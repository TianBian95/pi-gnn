dataset=ogbn-arxiv      #Datasets for evaluation: ogbn-arxiv
type=gcn          #Backbone: gcn, gat, sage
cor_type=uniform  #The type of label noise: uniform, flip
cor_prob=0.2      #The ratio of label noise: [0.0, 1]
epochs=400        #Training epochs
norm=10000
python3 ./model_bigdata/PI-GCNwopc.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs --norm $norm