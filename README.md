# TreePool

## Overview

The repository includes the source codes in our paper entitled "Improving Code Representation Learning via Multi-view Contrastive Graph Pooling for Abstract Syntax Tree". The model can be applied on code classification task w two dataset OJ-104 and OJ-DEEP according to our paper.  


## Usage

1. Download data from the data Source: https://sites.google.com/site/treebasedcnn/

2. Install all the dependent packages via pip according to your environment:
```
Python >= 3.8
pip install
pytorch=1.9.1
torch-cluster=1.5.9
torch-geometric=2.0.2
torch-scatter=2.0.8
torch-sparse=0.6.11
```

3. Modify the config json file to train and test the model.

## Directory Structure

```
└─ dataset
   ├─ OJDatasetLoader:for loading OJ dataset
   ├─ OJDeepDatasetLoader:for loading OJ-DEEP dataset

└─ layers
   ├─ GNN_node: Graph Neural Network Layer
   ├─ GNN_virtual_node: Graph Neural Network Layer with virtual node
   ├─ encoders: Node and edge encoders module
   ├─ gat_layer: Graph Attention Layer
   ├─ gcn_layer: Graph Convolution Layer
   ├─ gin_layer: Graph Isomorphism Network Layer
   ├─ graphsage_layer: GraphSAGE Layer
   ├─ mlp_layer: Multi-Layer Perceptron Layer

└─ models
   ├─ model: Model definition

└─ pooling
   ├─ lastreadout_layer: Last Readout Layer
   ├─ pooling_layer: Pooling Layer, main inplement of TreePool

└─ utils
   └─ file: File operation utilities
   └─ parameter: Parameter utilities
   └─ train_util:Training utilities

└─ train_parallel: main function of train, val and test procedure
```