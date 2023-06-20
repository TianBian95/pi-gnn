dataset=ogbn-arxiv      #Datasets for evaluation: ogbn-arxiv
type=gcn          #Backbone: gcn, gat, sage
cor_type=uniform  #The type of label noise: uniform, flip
cor_prob=0.2      #The ratio of label noise: [0.0, 1]
epochs=400        #Training epochs
norm=10000
type2=gcn
miself=False
start_epoch=50
python3 ./model_bigdata/PI-GCN.py --dataset $dataset --type $type --type2 $type2 --miself $miself --start_epoch $start_epoch \
--corruption_type $cor_type --corruption_prob $cor_prob --epochs $epochs --norm $norm