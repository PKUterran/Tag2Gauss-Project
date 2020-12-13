import numpy as np
import torch
import torch.optim as optim
from itertools import chain
from sklearn.metrics.classification import f1_score

from data.bilibili1.reader import read
from .net import TagCentral, GAT, Classifier

H_DIM = 64
EPSILON = 1.0
SEED = 0
PART = 0.7


def train(use_cuda=False):
    # seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # data
    nf, tf, adj, label = read()
    print(nf.shape)
    print(tf.shape)
    print(adj.shape)
    nf = torch.tensor(nf, dtype=torch.float32)
    tf = torch.tensor(tf, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    if use_cuda:
        nf, tf, adj, label = nf.cuda(), tf.cuda(), adj.cuda(), label.cuda()

    n_node = nf.shape[0]
    indices = np.random.permutation(range(n_node))
    train_mask = indices[:int(n_node * PART)]
    test_mask = indices[int(n_node * PART):]
    train_label, test_label = label[train_mask, :], label[test_mask, :]

    f_dim = nf.shape[1]
    t_dim = tf.shape[1]
    l_dim = label.shape[1]

    # model
    tag_central = TagCentral(t_dim, H_DIM, use_cuda=use_cuda)
    # gat = GAT(f_dim, H_DIM, use_cuda=use_cuda)
    gat = GAT(f_dim + t_dim, H_DIM, use_cuda=use_cuda)
    classifier = Classifier(H_DIM, l_dim, use_cuda=use_cuda)
    if use_cuda:
        tag_central.cuda()
        gat.cuda()
        classifier.cuda()
    optimizer = optim.Adam(chain(tag_central.parameters(), gat.parameters(), classifier.parameters()), lr=1e-3)

    # train
    for i in range(1000):
        epoch = i + 1

        tag_central.train()
        gat.train()
        classifier.train()
        optimizer.zero_grad()
        central = tag_central.forward(tf)
        hf = gat.forward(torch.cat([nf, tf], dim=1), adj)
        lb = classifier.forward(hf)
        train_lb, test_lb = lb[train_mask], lb[test_mask]

        dis_loss = torch.mean(torch.relu(torch.sum((hf - central) ** 2, dim=1) - EPSILON))
        ce_loss = torch.mean(torch.sum((train_lb - train_label) ** 2, dim=1))
        loss = dis_loss * 0.0 + ce_loss
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            train_f1 = f1_score(train_lb.argmax(dim=1).cpu().detach().numpy(),
                                train_label.argmax(dim=1).cpu().detach().numpy(),
                                average='macro')
            test_f1 = f1_score(test_lb.argmax(dim=1).cpu().detach().numpy(),
                               test_label.argmax(dim=1).cpu().detach().numpy(),
                               average='macro')
            print('#################')
            print('EPOCH:', epoch)
            print('DIS_LOSS:', dis_loss.item())
            print('CE_LOSS:', ce_loss.item())
            print('TRAIN_F1:', train_f1)
            print('TEST_F1:', test_f1)
