import sys,os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import argparse
import scipy.sparse as sp
import os
import numpy as np
import random
from tqdm import tqdm
from torch.nn import Linear as Lin
from utils.corrupte import uniform_mix_C_revised, flip_labels_C
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import NeighborSampler

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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dc = InnerProductDecoder()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x # Target nodes are always placed first.
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        x_product = self.dc(x[:size[1]])
        return x[:size[1]].log_softmax(dim=-1),x_product

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x[:size[1]].cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()

        return x_all

class GatNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels,
                                  heads))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(dataset.num_features, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

        self.dc = InnerProductDecoder()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        x_product = self.dc(x)
        return x.log_softmax(dim=-1),x_product

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = x + self.skips[i](x_target)
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class Sage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dc = InnerProductDecoder()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        x_product = self.dc(x)
        return x.log_softmax(dim=-1),x_product

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', default='ogbn-arxiv', type=str)
    args.add_argument('--epochs', default=400, type=int)
    args.add_argument('--seed', default=1, type=int)
    args.add_argument('--norm', default=10000, type=float)
    args.add_argument('--gpu', default=0, type=int)
    args.add_argument('--vanilla', default=0, type=int)
    args.add_argument('--plot', default=0, type=int)
    args.add_argument('--sparse', default=0, type=int)
    args.add_argument('--start_epoch', default=50, type=int)
    args.add_argument('--type', default='gcn', type=str)
    args.add_argument('--type2', default='gcn', type=str)
    args.add_argument('--miself',default=False,type=bool)
    # args.add_argument('--multi_gpu', default=0, type=int)
    args.add_argument('--corruption_prob',type=float,default=0,help='The ratio of label noise')
    args.add_argument('--corruption_type',type=str,default='uniform',help='The type of label noise:uniform,flip')
    args = args.parse_args()

    # print(args)
    # if not args.multi_gpu:
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.set_device(int(args.gpu))
    # training settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.dataset=='ogbn-arxiv':
        dataset = PygNodePropPredDataset('ogbn-arxiv', 'data/ogbn-arxiv/')
        evaluator = Evaluator(name='ogbn-arxiv')
    else:
        raise Exception("Dataset not implemented!")

    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']

    # if not args.multi_gpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.type == 'gcn':
        model = Net(dataset.num_features, 16, dataset.num_classes, num_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                       sizes=[15, 10], batch_size=1024,
                                       shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                          batch_size=4096, shuffle=False,
                                          num_workers=12)
    elif args.type == 'gat':
        model = GatNet(dataset.num_features, 8, dataset.num_classes, num_layers=2,
            heads=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                       sizes=[15, 10], batch_size=512,
                                       shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                          batch_size=1024, shuffle=False,
                                          num_workers=12)
    elif args.type == 'sage':
        model = Sage(dataset.num_features, 64, dataset.num_classes, num_layers=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                       sizes=[15, 10], batch_size=4096,
                                       shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                          batch_size=4096, shuffle=False,
                                          num_workers=12)

    if args.type2 == 'gcn':
        model_mi = Net(dataset.num_features, 16, dataset.num_classes, num_layers=2).to(device)
        optimizer_mi = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    elif args.type2 == 'gat':
        model_mi = GatNet(dataset.num_features, 8, dataset.num_classes, num_layers=2,
            heads=8).to(device)
        optimizer_mi = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    elif args.type2 == 'sage':
        model_mi = Sage(dataset.num_features, 64, dataset.num_classes, num_layers=2).to(device)
        optimizer_mi = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    if args.corruption_type == 'uniform':
        transition_matrix = uniform_mix_C_revised(args.corruption_prob, dataset.num_classes)
    else:
        transition_matrix = flip_labels_C(args.corruption_prob, dataset.num_classes, seed=args.seed)
    label = []

    for label_i in y:
        data1 = np.random.choice(dataset.num_classes, p=transition_matrix[label_i])
        label.append(data1)
    ##end##
    label = np.array(label)
    label = torch.from_numpy(label).cuda()

    all_acc = []
    for iter in range(10):
        # if not args.multi_gpu:
        args.seed=iter
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)


        if args.sparse:
            train_edges = torch.transpose(data.adj_t.to_dense(), 0, 1).cpu().data.numpy()
        else:
            train_edges = torch.transpose(data.edge_index, 0, 1).cpu().data.numpy()

        data_context = np.ones(train_edges.shape[0])
        # Re-build adj matrix
        adj_train = sp.csr_matrix((data_context,
                                   (train_edges[:, 0], train_edges[:, 1])),
                                  shape=(len(y), len(y)))
        adj_train = (adj_train + adj_train.T) / 2
        adj_label = adj_train + sp.eye(adj_train.shape[0])
        pos_weight = torch.Tensor([float(len(y) * len(y) - len(data_context)) / len(data_context)]).cuda()
        if args.norm == 10000:
            norm = len(data.y) * len(data.y) / float((len(data.y) * len(data.y) - len(data_context)) * 2)
        else:
            norm = args.norm

        best_val_acc = 0
        cor_test_acc = 0


        for epoch in range(args.epochs):
            model_mi.train()
            loss_mi = 0
            total_correct = 0
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                optimizer_mi.zero_grad()
                out_mi, out_product_mi = model_mi(x[n_id], adjs)
                id = n_id[:batch_size]
                row_adj_label = adj_label[id]
                row_col_adj_label = row_adj_label[:, id]
                labels_context = torch.FloatTensor(row_col_adj_label.toarray()).to(device)
                loss_mi_temp = norm * F.binary_cross_entropy_with_logits(
                    out_product_mi, labels_context,pos_weight=pos_weight)
                loss_mi_temp.backward()
                optimizer_mi.step()
                loss_mi = loss_mi + float(loss_mi_temp)
                total_correct += int(out_mi.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
            loss_mi = loss_mi / len(train_loader)
            approx_acc = total_correct / len(train_loader)
            # print(f'mi_Loss: {loss_mi:.4f}, Train: {approx_acc:.4f}')

        for epoch in range(args.epochs):
            total_loss = 0
            total_correct = 0
            model.train()
            pbar = tqdm(total=train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()
                out, out_product = model(x[n_id], adjs)
                loss = F.nll_loss(out, label[n_id[:batch_size]])
                out_mi, out_product_mi = model_mi(x[n_id], adjs)
                id = n_id[:batch_size]
                row_adj_label = adj_label[id]
                row_col_adj_label = row_adj_label[:, id]
                labels_context = torch.FloatTensor(row_col_adj_label.toarray()).to(device)
                pos_position = labels_context.view(-1).bool()
                neg_position = (1 - labels_context).view(-1).bool()
                mask = torch.zeros_like(out_product_mi).view(-1).cuda()
                mask[pos_position] = torch.sigmoid(out_product_mi).view(-1)[pos_position]
                mask[neg_position] = 1 - torch.sigmoid(out_product_mi).view(-1)[neg_position]
                mask = mask.view(labels_context.size(0), labels_context.size(1))
                if epoch > args.start_epoch:
                    temp_loss = norm * (F.binary_cross_entropy_with_logits(
                        out_product,labels_context,pos_weight=pos_weight, reduction='none') * mask.detach()).mean()
                else:
                    temp_loss = norm * F.binary_cross_entropy_with_logits(
                        out_product,labels_context,pos_weight=pos_weight)
                train_loss = loss + temp_loss
                train_loss.backward()
                optimizer.step()
                total_loss = total_loss + float(train_loss)
                total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
                pbar.update(batch_size)
            pbar.close()
            total_loss = total_loss / len(train_loader)
            approx_acc = total_correct / train_idx.size(0)

            model.eval()
            out = model.inference(x)
            y_true = y.cpu().unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)
            train_acc = evaluator.eval({
                'y_true': y_true[split_idx['train']],
                'y_pred': y_pred[split_idx['train']],
            })['acc']
            val_acc = evaluator.eval({
                'y_true': y_true[split_idx['valid']],
                'y_pred': y_pred[split_idx['valid']],
            })['acc']
            test_acc = evaluator.eval({
                'y_true': y_true[split_idx['test']],
                'y_pred': y_pred[split_idx['test']],
            })['acc']
            # print(f'Loss: {total_loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            #       f'Test: {test_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                cor_test_acc = test_acc

        all_acc.append(cor_test_acc)
    print('{:.3f}({:.2f})'.format(np.mean(all_acc), np.std(all_acc)))
