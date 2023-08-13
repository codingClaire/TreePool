import torch
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import json
import random
import os
import copy
import datetime

from dataset.OJDatasetLoader import extract_OJdataset, extract_small_OJdataset
from dataset.OJDatasetLoader import OJDatasetLoader
from dataset.OJDeepDatasetLoader import extract_deep_OJdataset

from utils.parameter import check_parameter
from utils.file import save_csv_with_configname, save_npy_with_name
from utils.train_util import load_epoch

from models.model import Model


bcls_criterion = torch.nn.BCEWithLogitsLoss()
mcls_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()


def train(model, device, loader, optimizer):
    model.train()
    losses = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration", mininterval=60)):
        pred, model_loss = model(batch)
        optimizer.zero_grad()
        y_arr = [b.y for b in batch]
        y_arr = torch.cat(y_arr).to(device)
        loss = 0
        loss += mcls_criterion(pred.to(torch.float32), y_arr)
        loss += sum(model_loss).item()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def eval(model, loader):
    model.eval()
    correct_num = 0
    total_num = 0
    for _, batch in enumerate(tqdm(loader, desc="Iteration", mininterval=60)):
        with torch.no_grad():
            pred, _ = model(batch)
        y_true = [b.y for b in batch]
        y_pred = torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu()
        for i in range(len(y_pred)):
            if(y_true[i] == y_pred[i]):
                correct_num += 1
        total_num += len(y_true)
    acc = correct_num / total_num
    return acc


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main(net_parameters):
    net_parameters = check_parameter(net_parameters)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_parameters["device"] = device

    seed_everything(net_parameters["seed"])

    # get train/valid/test set
    if "deep" in net_parameters and net_parameters["deep"] == True:
        # deep or not
        print("depth ratio is:", net_parameters["depth_ratio"])
        train_set, valid_set, test_set, type_num = extract_deep_OJdataset(net_parameters["data_path"], net_parameters["valid_ratio"], net_parameters["num_vocab"],
                                                                          net_parameters["depth_augment"], net_parameters["depth_group"], net_parameters["depth_ratio"])
        train_set = OJDatasetLoader.load_dataset(train_set)
        valid_set = OJDatasetLoader.load_dataset(valid_set)
        test_set = OJDatasetLoader.load_dataset(test_set)
    elif "debug" not in net_parameters or net_parameters["debug"] == False:
        train_set, valid_set, test_set, type_num = extract_OJdataset(net_parameters["data_path"], net_parameters["valid_ratio"], net_parameters["num_vocab"],
                                                                     net_parameters["depth_augment"], net_parameters["depth_group"])
        train_set = OJDatasetLoader.load_dataset(train_set)
        valid_set = OJDatasetLoader.load_dataset(valid_set)
        test_set = OJDatasetLoader.load_dataset(test_set)
    else:
        train_set, valid_set, test_set, type_num = extract_small_OJdataset(net_parameters["data_path"], net_parameters["valid_ratio"], net_parameters["num_vocab"],
                                                                           net_parameters["depth_augment"], net_parameters["depth_group"])
        train_set = OJDatasetLoader.load_dataset(train_set)
        valid_set = OJDatasetLoader.load_dataset(valid_set)
        test_set = OJDatasetLoader.load_dataset(test_set)

    train_loader = DataListLoader(
        train_set,
        batch_size=net_parameters["batch_size"],
        shuffle=True,
        num_workers=net_parameters["num_workers"],
    )
    valid_loader = DataListLoader(
        valid_set,
        batch_size=net_parameters["batch_size"],
        shuffle=False,
        num_workers=net_parameters["num_workers"],
    )
    test_loader = DataListLoader(
        test_set,
        batch_size=net_parameters["batch_size"],
        shuffle=False,
        num_workers=net_parameters["num_workers"],
    )

    # meta info add
    net_parameters["num_nodetypes"] = type_num
    net_parameters["max_depth"] = 20
    net_parameters["in_dim"] = 0
    net_parameters["edge_dim"] = 1
    net_parameters["num_tasks"] = 104

    # initialize model
    model = Model(net_parameters)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[0, 1])

    optimizer_state_dict = None
    if net_parameters["load"] > 0:
        model_state_dict, optimizer_state_dict = load_epoch(
            net_parameters["model_path"], net_parameters["load"]
        )
        model.load_state_dict(model_state_dict)

    model = model.to(device)
    print("="*30)
    print(model)
    print("="*30)
    optimizer = optim.Adam(
        model.parameters(), lr=net_parameters["learning_rate"]
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    train_curve, valid_curve, test_curve = [], [], []
    loss_list = []
    for epoch in range(net_parameters["load"] + 1, net_parameters["epochs"] + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss = train(model, device, train_loader, optimizer)
        loss_list.append(loss)
        print("Evaluating...")

        train_acc = eval(
            model, train_loader) if net_parameters["eval_train"] == True else 0
        valid_acc = eval(model, valid_loader)
        test_acc = eval(model, test_loader)
        print({"Train": train_acc, "Validation": valid_acc, "Test": test_acc})
        # check continue fials
        if len(valid_curve) == 0 or valid_curve[-1] < valid_acc:
            # will be update  (larger better)
            continues_fials = net_parameters["continues_fials"]
        else:
            continues_fials -= 1

        train_curve.append(train_acc)
        valid_curve.append(valid_acc)
        test_curve.append(test_acc)
        best_val_epoch = np.argmax(np.array(valid_curve))
        print("current_best:")
        print("Validation score: {}".format(
            round(valid_curve[best_val_epoch], 4)))
        print("Test score: {}".format(round(test_curve[best_val_epoch], 4)))

        if (net_parameters["save_every_epoch"] and epoch % net_parameters["save_every_epoch"] == 0):
            tqdm.write("saving to epoch.%04d.pth" % epoch)
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                os.path.join(
                    net_parameters["model_path"], "epoch.%04d.pth" % epoch),
            )
        print(continues_fials)
        if continues_fials == 0:
            print(
                f"The performance of the model has not been improved by consecutive {net_parameters['continues_fials']} epoch, early stop")
            break
    ############# end epoch #####################
    best_val_epoch = np.argmax(np.array(valid_curve))
    print("Finished training!")
    print("Best validation:")
    print("Validation score: {}".format(valid_curve[best_val_epoch]))
    print("Test score: {}".format(test_curve[best_val_epoch]))

    result = {
        "Train": train_curve[best_val_epoch],
        "Val": valid_curve[best_val_epoch],
        "Test": test_curve[best_val_epoch],
    }
    return result, loss_list


def repeat_experiment():
    configs = [
        "test.json"
    ]
    for config in configs:
        config_name = config.split("/")[-1].split(".")[0]
        print(config_name)
        with open(config) as f:
            total_parameters = json.load(f)
        total_parameters["config_name"] = config_name
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
        for seed in total_parameters["seed"]:
            for pooling in total_parameters["pooling"]:
                for emb_dim in total_parameters["emb_dim"]:
                    for readout in total_parameters["final_readout"]:
                        for lr in total_parameters["learning_rate"]:
                            print("current seed: ", seed,
                                  "current pooling:", pooling,
                                  "final_readout:", readout,
                                  "emb_dim:", emb_dim,
                                  "learning_rate:", lr
                                  )
                            net_parameters = copy.deepcopy(
                                total_parameters)
                            net_parameters["seed"] = seed
                            net_parameters["pooling"] = pooling
                            net_parameters["emb_dim"] = emb_dim
                            net_parameters["learning_rate"] = lr
                            net_parameters["final_readout"] = readout

                            net_parameters["time"] = datetime.datetime.now().strftime(
                                '%Y-%m-%d_%H:%M')

                            net_parameters["model_path"] = net_parameters["model_path"] + "/" + \
                                net_parameters["config_name"] + \
                                "__" + str(net_parameters["time"])
                            result, loss_list = main(net_parameters)
                            net_parameters.update(result)
                            save_csv_with_configname(
                                net_parameters, "oj/result/" + net_parameters["date"], config_name)
                            save_npy_with_name(
                                loss_list, net_parameters, "oj/loss", config_name)
                            print(datetime.datetime.now().strftime(
                                '%Y-%m-%d %H:%M'))


if __name__ == "__main__":
    repeat_experiment()
