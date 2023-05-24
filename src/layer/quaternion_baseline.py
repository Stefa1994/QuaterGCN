'''
Quaternion Graph Neural Network
Copy from: https://github.com/daiquocnguyen/QGNN/blob/master/QGNN_pytorch/q4gnn.py
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1)//4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGNNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNNLayer_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

'''Quaternion graph neural networks! QGNN layer for node and graph classification tasks!'''
class QGNNLayer(Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True, act=F.relu):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff 
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #
        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, double_type_used_in_graph=False):
        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix
        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)
    


'''Quaternion graph neural network! 2-layer!
    For link prediction
'''
class QGNN_Link(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(QGNN_Link, self).__init__()
        self.dropout = dropout
        self.q4gnn1 = QGNNLayer(nfeat, nhid, dropout=dropout)
        self.q4gnn2 = QGNNLayer(nhid, nhid, dropout=dropout) 
        self.linear = nn.Linear(nhid*2, nclass) # prediction layer

    def forward(self, x, adj, index):
        x = self.q4gnn1(x, adj)
        x = self.q4gnn2(x, adj)
        dim = x.size(1)//4
        r, i, j, k = torch.split(x, [dim, dim, dim, dim], dim=1)
        x = torch.cat((r[index[:,0]], r[index[:,1]], i[index[:,0]], i[index[:,1]], j[index[:,0]], j[index[:,1]], k[index[:,0]], k[index[:,1]]), axis=-1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        #x = torch.cat((x[index[:,0]], x[index[:,1]]), axis=-1)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)
    

'''Quaternion graph neural network! 2-layer Q4GNN!'''
#class QGNN_node(torch.nn.Module):
#    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
#        super(QGNN_node, self).__init__()
#        self.q4gnn1 = QGNNLayer(nfeat, nhid, dropout=dropout) 
#        self.q4gnn2 = QGNNLayer(nhid, nclass, dropout=dropout, quaternion_ff=False, act=lambda x:x) # prediction layer
#
#    def forward(self, x, adj):
#        x = self.q4gnn1(x, adj)
#        x = self.q4gnn2(x, adj)
#        return F.log_softmax(x, dim=1)


'''Quaternion graph neural network! 2-layer!
    For link prediction
'''
class QGNN_node(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(QGNN_node, self).__init__()
        self.dropout = dropout
        self.q4gnn1 = QGNNLayer(nfeat, nhid, dropout=dropout)
        self.q4gnn2 = QGNNLayer(nhid, nhid, dropout=dropout) 
        self.Conv = nn.Conv1d(nhid, nclass, kernel_size=1) # prediction layer

    def forward(self, x, adj):
        x = self.q4gnn1(x, adj)
        x = self.q4gnn2(x, adj)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)
        x = x.unsqueeze(0)
        x = x.permute((0,2,1))
        x = self.Conv(x)
        x = x.permute((0,2,1)).squeeze()


        return F.log_softmax(x, dim=1)

