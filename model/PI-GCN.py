import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid,WikiCS
import argparse
import scipy.sparse as sp
import os
import numpy as np
import random
from utils.corrupte import uniform_mix_C_revised, flip_labels_C

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora', type=str)
args.add_argument('--epochs', default=400, type=int)
args.add_argument('--seed', default=1, type=int)
args.add_argument('--norm', default=10000, type=float)
args.add_argument('--gpu', default=0, type=int)
args.add_argument('--vanilla', default=0, type=int)
args.add_argument('--plot', default=0, type=int)
args.add_argument('--sparse', default=0, type=int)
args.add_argument('--start_epoch', default=200, type=int)
args.add_argument('--type', default='gcn', type=str)
args.add_argument('--type2', default='gcn', type=str)
args.add_argument('--miself',default=False,type=bool)
# args.add_argument('--multi_gpu', default=0, type=int)
args.add_argument('--corruption_prob',type=float,default=0.4,help='The ratio of label noise')
args.add_argument('--corruption_type',type=str,default='uniform',help='The type of label noise:uniform,flip')
args = args.parse_args()



# print(args)
# if not args.multi_gpu:
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.set_device(int(args.gpu))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# os.environ['PYTHONHASHSEED'] = str(args.seed)



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, act=lambda x: x):
        super(InnerProductDecoder, self).__init__()
        # self.dropout = dropout
        self.act = act

    def forward(self, z):
        # z = F.dropout(z, self.dropout, training=self.training)
        # breakpoint()
        adj = self.act(torch.mm(z, z.t()))
        return adj

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.dc = InnerProductDecoder()

    def forward(self, data):
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index


        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x_product = self.dc(x)
        # breakpoint()
        return F.log_softmax(x, dim=1), x_product

class Net_batched(torch.nn.Module):
    def __init__(self):
        super(Net_batched, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
        self.dc = InnerProductDecoder()

    def forward(self, data):
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        # breakpoint()
        return F.log_softmax(x, dim=1), x

class GatNet(torch.nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)
        self.dc = InnerProductDecoder()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        x_product = self.dc(x)
        return F.log_softmax(x, dim=-1), x_product

class GatNet_batched(torch.nn.Module):
    def __init__(self):
        super(GatNet_batched, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)
        self.dc = InnerProductDecoder()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=-1), x

class Sage(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(Sage, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)
        self.dc = InnerProductDecoder()

    def forward(self, data):
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        x_product = self.dc(x)
        return F.log_softmax(x, dim=1), x_product

class Sage_batched(torch.nn.Module):
    def __init__(self, features, hidden, classes):
        super(Sage_batched, self).__init__()
        self.sage1 = SAGEConv(features, hidden)
        self.sage2 = SAGEConv(hidden, classes)
        self.dc = InnerProductDecoder()

    def forward(self, data):
        if args.sparse:
            x, edge_index = data.x, data.adj_t
        else:
            x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1), x


if args.dataset == 'cora':
    dataset = Planetoid(root='data/', name='Cora')
elif args.dataset == 'cite':
    dataset = Planetoid(root='data/', name='CiteSeer')
elif args.dataset == 'pub':
    dataset = Planetoid(root='data/', name='PubMed')
elif args.dataset == 'wiki':
    dataset = WikiCS(root='data/WikiCS/')
else:
    raise Exception("Dataset not implemented!")


# if not args.multi_gpu:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_acc = []
for iter in range(10):
    # if not args.multi_gpu:
    args.seed=iter
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.type == 'gcn':
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.type == 'gat':
        model = GatNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif args.type == 'sage':
        model = Sage(dataset.num_node_features, 64, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    if args.type2 == 'gcn':
        model_mi = Net().to(device)
        optimizer_mi = torch.optim.Adam(model_mi.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.type2 == 'gat':
        model_mi = GatNet().to(device)
        optimizer_mi = torch.optim.Adam(model_mi.parameters(), lr=0.005, weight_decay=5e-4)
    elif args.type2 == 'sage':
        model_mi = Sage(dataset.num_node_features, 64, dataset.num_classes).to(device)
        optimizer_mi = torch.optim.Adam(model_mi.parameters(), lr=0.01, weight_decay=5e-4)

    data = dataset[0].to(device)


    # import ipdb; ipdb.set_trace()
    ## label noise##
    if args.corruption_type == 'uniform':
        transition_matrix = uniform_mix_C_revised(args.corruption_prob, dataset.num_classes)
    else:
        transition_matrix = flip_labels_C(args.corruption_prob, dataset.num_classes, seed=args.seed)
    label = []

    if args.dataset =='wiki':
        class_wise_dict = {}
        for index in range(len(data.y)):
            key = int(data.y[index].cpu().data.numpy())
            if key not in class_wise_dict:
                class_wise_dict[key] = [index]
            else:
                class_wise_dict[key].append(index)
        train_mask = []
        val_mask = []
        test_mask = []
        for index in range(dataset.num_classes):
            train_mask += class_wise_dict[index][0:20]
            val_mask += class_wise_dict[index][20:40]
            test_mask += class_wise_dict[index][40:]

        data.train_mask = torch.zeros(len(data.y)).cuda()
        data.train_mask[train_mask] = 1
        data.val_mask = torch.zeros(len(data.y)).cuda()
        data.val_mask[val_mask] = 1
        data.test_mask = torch.zeros(len(data.y)).cuda()
        data.test_mask[test_mask] = 1
        data.train_mask = data.train_mask.bool()
        data.val_mask = data.val_mask.bool()
        data.test_mask = data.test_mask.bool()


    for label_i in data.y[data.train_mask]:
        data1 = np.random.choice(dataset.num_classes, p=transition_matrix[label_i])
        label.append(data1)
    ##end##
    label = np.array(label)
    label = torch.from_numpy(label).cuda()
    # breakpoint()
    # print(label)
    best_val_acc = 0
    cor_test_acc = 0

    # construct the adj label.
    # import ipdb; ipdb.set_trace()
    if args.sparse:
        train_edges = torch.transpose(data.adj_t.to_dense(), 0, 1).cpu().data.numpy()
    else:
        train_edges = torch.transpose(data.edge_index, 0, 1).cpu().data.numpy()

    data_context = np.ones(train_edges.shape[0])
    # Re-build adj matrix
    adj_train = sp.csr_matrix((data_context,
                               (train_edges[:, 0], train_edges[:, 1])),
                              shape=(len(data.y), len(data.y)))
    # breakpoint()
    adj_train = (adj_train + adj_train.T) / 2
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray()).cuda()
    pos_weight = torch.Tensor([float(len(data.y) * len(data.y) - len(data_context)) / len(data_context)]).cuda()
    #
    for epoch in range(args.epochs):
        loss_context = 0
        model.train()
        model_mi.train()
        optimizer.zero_grad()
        optimizer_mi.zero_grad()
        if args.type == 'gcn' or args.type == 'sage':
            out, out_product = model(data)
        if args.type2 == 'gcn' or args.type2 == 'sage':
            out_mi, out_product_mi = model_mi(data)
        if args.type == 'gat':
            if args.sparse:
                out, out_product = model(data.x, data.adj_t)
            else:
                out, out_product = model(data.x, data.edge_index)
        if args.type2 == 'gat':
            out_mi, out_product_mi = model_mi(data.x, data.edge_index)
        # out, out_product = model(data)
        loss = F.nll_loss(out[data.train_mask], label)#data.y[data.train_mask])
        labels_context = adj_label
        if args.norm == 10000:
            norm = len(data.y) * len(data.y) / float((len(data.y) * len(data.y) - len(data_context)) * 2)
        else:
            norm = args.norm
        # breakpoint()
        # implement batched version in order to solve the memory issues.
        if args.dataset == 'auth' or args.dataset == 'ama' or args.dataset == 'corafull' or args.dataset == 'reddit':
            batch_size = 5000
            batches = len(out_product) // batch_size + 1
            out_product_t = out_product.t()
            out_product_t_mi = out_product_mi.t()

            loss_mi = 0
            for index_mi in range(batches):
                predictions_mi = \
                    torch.mm(out_product_mi[index_mi * batch_size:index_mi * batch_size + batch_size],
                             out_product_t_mi)
                loss_mi_temp = norm * F.binary_cross_entropy_with_logits(
                    predictions_mi, labels_context[index_mi * batch_size:index_mi * batch_size + batch_size],
                    pos_weight=pos_weight)
                loss_mi += loss_mi_temp
                if index_mi == 0:
                    out_product_mi_new = predictions_mi
                else:
                    out_product_mi_new = torch.cat((out_product_mi_new, predictions_mi), 0)
                del predictions_mi
            loss_mi /= batches
            loss_mi.backward()
            optimizer_mi.step()
            mask = torch.zeros_like(out_product_mi_new).view(-1).cuda()
            pos_position = labels_context.view(-1).bool()
            neg_position = (1 - labels_context).view(-1).bool()
            mask[pos_position] = torch.sigmoid(out_product_mi_new).view(-1)[pos_position]
            mask[neg_position] = 1 - torch.sigmoid(out_product_mi_new).view(-1)[neg_position]
            mask = mask.view(labels_context.size(0), labels_context.size(1))

            for index in range(batches):
                predictions = \
                    torch.mm(out_product[index * batch_size:index * batch_size + batch_size], out_product_t)
                if epoch > args.start_epoch:
                    temp_loss = norm * F.binary_cross_entropy_with_logits(
                        predictions,
                        labels_context[index*batch_size:index*batch_size+batch_size],
                        pos_weight=pos_weight, reduction='none') * mask[index*batch_size:index*batch_size+batch_size].detach()
                    temp_loss = temp_loss.mean()
                else:
                    temp_loss = norm * F.binary_cross_entropy_with_logits(
                        predictions,
                        labels_context[index * batch_size:index * batch_size + batch_size],
                        pos_weight=pos_weight)

                loss_context += temp_loss
                del predictions
            loss_context /= batches
        else:
            loss_mi = norm * F.binary_cross_entropy_with_logits(
                out_product_mi, labels_context, pos_weight=pos_weight)
            loss_mi.backward()
            optimizer_mi.step()
            mask = torch.zeros_like(out_product_mi).view(-1).cuda()
            pos_position = labels_context.view(-1).bool()
            neg_position = (1 - labels_context).view(-1).bool()
            if args.miself == False:
                mask[pos_position] = torch.sigmoid(out_product_mi).view(-1)[pos_position]
                mask[neg_position] = 1 - torch.sigmoid(out_product_mi).view(-1)[neg_position]
                mask = mask.view(labels_context.size(0), labels_context.size(1))
            else:
                predictions = out_product
                mask[pos_position] = torch.sigmoid(predictions).view(-1)[pos_position]
                mask[neg_position] = 1 - torch.sigmoid(predictions).view(-1)[neg_position]
                mask = mask.view(labels_context.size(0), labels_context.size(1))

            if epoch > args.start_epoch:
                if epoch == args.epochs - 1 and args.plot:
                    import pandas as pd
                    import seaborn as sns
                    import matplotlib
                    matplotlib.use('AGG')
                    import matplotlib.pyplot as plt
                    plt.style.use('seaborn-dark')
                    out_product_mi = out_product_mi.view(-1)
                    out_product_mi[neg_position] = 1 - out_product_mi[neg_position]
                    data_plot = pd.Series(out_product_mi.cpu().data.numpy())
                    p1 = sns.kdeplot(data_plot, shade=True, color="b", label='MI')
                    plt.legend()
                    # plt.savefig('motivation_1.jpg', dpi=250)
                loss_context = norm * (F.binary_cross_entropy_with_logits(
                    out_product, labels_context, pos_weight=pos_weight, reduction='none') * mask.detach()).mean()
            else:
                loss_context = norm * F.binary_cross_entropy_with_logits(
                    out_product, labels_context, pos_weight=pos_weight)
        if args.vanilla:
            loss_context = 0
        loss += loss_context
        loss.backward()
        optimizer.step()

        model.eval()
        model_mi.eval()
        if args.type == 'gcn' or args.type=='sage':
            _, pred = model(data)[0].max(dim=1)
        elif args.type == 'gat':
            if args.sparse:
                _, pred = model(data.x, data.adj_t)[0].max(dim=1)
            else:
                _, pred = model(data.x, data.edge_index)[0].max(dim=1)
        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        val_acc = correct / int(data.val_mask.sum())

        correct_test = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
        test_acc = correct_test / int(data.test_mask.sum())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cor_test_acc = test_acc

        # if epoch % 50 == 0:
        #     print('Epoch: ', epoch)
        #     print('Training Loss this epoch: ', loss.item())
        #     print('MI Loss this epoch: ', loss_mi.item())

    all_acc.append(cor_test_acc)
print('{:.3f}({:.2f})'.format(np.mean(all_acc), np.std(all_acc)))