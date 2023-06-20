dataset=pub      #Datasets for evaluation: cora, cite, pub, wiki
type=gcn          #Backbone: gcn, gat, sage
cor_type=uniform  #The type of label noise: uniform, flip
cor_prob=0.2      #The ratio of label noise: [0.0, 1]
epochs=400        #Training epochs
norm=10000
batchsize=5000
python3 ./model/PI-GCNwopc.py --dataset $dataset --type $type --corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs --norm $norm --batchsize $batchsize