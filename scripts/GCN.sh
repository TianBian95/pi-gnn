dataset=wiki      #Datasets for evaluation: cora, cite, pub, wiki
type=gcn          #Backbone: gcn, gat, sage
cor_type=uniform  #The type of label noise: uniform, flip
cor_prob=0.2      #The ratio of label noise: [0.0, 1]
epochs=400        #Training epochs
python3 ./model/GCN.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs