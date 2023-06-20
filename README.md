# Noise-robust Graph Learning by Estimating and Leveraging Pairwise Interactions, TMLR, 06/2023.


## 1 Package Dependency
We provide a list of reference versions for all packages in the `requirements.txt`, `pip` can be used to install these packages:
```
pip3 install -r requirements.txt
```

## 2 Run the code
We provide some scripts in the `./scripts` folder to run our code. 
For example, you can run the PI-GNN model by the command `bash ./scripts/PI-GNN.sh`. Below, we will explain the meanings of different commands in these scripts in detail.

### 2.1 Variables
First, we need to define some common variables: 
```
dataset=cora      #Datasets for evaluation: cora, cite, pub, wiki, ogbn-arxiv
type=gcn          #Backbone: gcn, gat, sage
cor_type=uniform  #The type of label noise: uniform, flip
cor_prob=0.2      #The ratio of label noise: [0.0, 1]
epochs=400        #Training epochs
```
More variables will be defined in specific scripts.


### 2.2 Classical graph learning models
For classical graph learning models (e.g., GCN, GAT, SAGE), run:

`python3 ./model/GCN.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

For large-scale dataset, such as `ogbn-arxiv`, run:

`python3 ./model_bigdata/GCN.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

### 2.3 PI-GNN w/o pc (predictive confidence)
For PI-GNN without predictive confidence, i.e., PI-GNN trained with the node connectivity, run:

`python3 ./model/PI-GCNwopc.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

For large-scale dataset, such as `ogbn-arxiv`, run:

`python3 ./model_bigdata/PI-GCNwopc.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

### 2.4 PI-GNN
For PI-GNN, run:

`python3 ./model/PI-GCN.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

For large-scale dataset, such as `ogbn-arxiv`, run:

`python3 ./model_bigdata/PI-GCN.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs`

## 3 Dataset
When you execute the above code, the PyTorch Geometric package will automatically download the corresponding raw data in the `./data/$dataset/raw/` folder,
and save the processed data in the `./data/$dataset/processed/` folder. 
We also provide the raw data through [Google Drive](https://drive.google.com/drive/folders/1ByeLbAhRWVBgQIhfxW3lp_T1T_juhmbb?usp=sharing) in case of data version inconsistency or failure to download due to network problems.