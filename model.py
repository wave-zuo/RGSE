import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.en_s = nn.Linear(self.in_dim, self.hidden1_dim)
        self.en_s2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)
        self.en_s3 = nn.Linear(self.hidden2_dim, 2)

    def forward(self, x):
        h = F.leaky_relu(self.en_s(x), 0.1)
        h1 = F.leaky_relu(self.en_s2(h), 0.1)
        h2 = F.leaky_relu(self.en_s3(h1), 0.1)
        return h2


class MyModel_combine(nn.Module):
    def __init__(self, emb_size, inter_hidden1_dim, inter_hidden2_dim, second_in_dim, hidden1_dim, hidden2_dim, dataset='cora', ratio=0.1):
        super(MyModel_combine, self).__init__()
        self.emb = torch.from_numpy(np.load('data/' + dataset + '/ae_embed_nloss2'+str(ratio)+'.npy')).float().cuda()
        self.interactive1 = nn.Linear(emb_size, inter_hidden1_dim)
        self.interactive2 = nn.Linear(inter_hidden1_dim, inter_hidden2_dim)
        self.second_fc = nn.Linear(second_in_dim, inter_hidden2_dim)
        self.fc1 = nn.Linear(inter_hidden2_dim*1, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, 2)
        self.weight = nn.Parameter(torch.Tensor(2, 1, 1))
        nn.init.constant_(self.weight, 0.1)

    def forward(self, x1, x2, second_feats):
        x1 = self.emb[x1]
        x2 = self.emb[x2]
        x = x1 * x2
        x = F.leaky_relu(self.interactive1(x))
        x = F.leaky_relu(self.interactive2(x))
        second_feats = F.leaky_relu(self.second_fc(second_feats))
        fusion_feats = torch.stack([x, second_feats])
        x = torch.sum(fusion_feats * F.softmax(self.weight, dim=0), dim=0)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class GAElinear(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim):
        super(GAElinear, self).__init__()
        self.in_dim = in_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.en_s = nn.Linear(self.in_dim, self.hidden1_dim)
        self.en_s2 = nn.Linear(self.hidden1_dim, self.hidden2_dim)

    def forward(self, adj):
        h = F.leaky_relu(self.en_s(adj))
        h_s = F.leaky_relu(self.en_s2(h))
        rec_s = torch.sigmoid(torch.matmul(h_s, h_s.t()))
        return rec_s, h_s