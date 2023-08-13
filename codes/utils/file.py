import csv
import os
import numpy as np


def save_csv(params, file_dir):
    # 1. change dics
    print(params)
    pool_method = params["pooling"]
    dataset_name = params["dataset_name"]

    # update new parameters
    if(pool_method in params.keys()):
        for pool_param in params[pool_method].keys():
            new_key = pool_method + "_" + pool_param
            params[new_key] = params[pool_method][pool_param]

        del params[pool_method]
    print(params)

    sheet_title = params.keys()
    sheet_title = sorted(sheet_title)
    print(sheet_title)
    sheet_data = []
    for key in sheet_title:
        sheet_data.append(params[key])
    # 2. find the file
    file_name = dataset_name+"_"+pool_method+".csv"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if os.path.exists(os.path.join(file_dir, file_name)):
        action = "a"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
    else:
        action = "w"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
        writer.writerow(sheet_title)
    writer.writerow(sheet_data)
    csv_fp.close()


def save_csv_with_configname(params, file_dir, configname):
    # 1. change dics
    print(params)
    pool_method = params["pooling"]
    dataset_name = params["dataset_name"]

    # update new parameters
    if(pool_method in params.keys()):
        for pool_param in params[pool_method].keys():
            new_key = pool_method + "_" + pool_param
            params[new_key] = params[pool_method][pool_param]

        del params[pool_method]
    print(params)

    sheet_title = params.keys()
    sheet_title = sorted(sheet_title)
    print(sheet_title)
    sheet_data = []
    for key in sheet_title:
        sheet_data.append(params[key])
    # 2. find the file
    file_name = dataset_name+"_"+pool_method+"_["+configname+"].csv"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if os.path.exists(os.path.join(file_dir, file_name)):
        action = "a"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
    else:
        action = "w"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
        writer.writerow(sheet_title)
    writer.writerow(sheet_data)
    csv_fp.close()


def save_npy_with_name(loss_list, params, file_dir, config_name):
    file_dir = "loss/" + config_name
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file_name = file_dir + "/"+str(params["seed"])
    np.save(file_name, np.array(loss_list))
