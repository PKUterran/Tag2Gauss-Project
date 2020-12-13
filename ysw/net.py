import torch
import torch.nn as nn


class TagCentral(nn.Module):
    def __init__(self, t_dim, h_dim, use_cuda=False):
        super(TagCentral, self).__init__()
        self.use_cuda = use_cuda
        self.central = nn.Linear(t_dim, h_dim)

    def forward(self, t: torch.Tensor):
        t = t / torch.sum(t, dim=1, keepdim=True)
        # print('t:', t.cpu().detach().numpy())
        c = self.central(t)
        return c


class AttentiveLayer(nn.Module):
    def __init__(self, h_dim, use_cuda=False):
        super(AttentiveLayer, self).__init__()
        self.use_cuda = use_cuda

        self.align = nn.Linear(h_dim, 1)
        self.al_act = nn.Softmax(dim=-1)
        self.attend = nn.Linear(h_dim, h_dim)
        self.at_act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert torch.max(adj) < 1 + 1e-5 and torch.min(adj) > -1e-5
        adj = (-adj + 1) * -1e6 + adj * self.align(x).reshape([-1])
        adj: torch.Tensor = self.al_act(adj)
        # print(adj.cpu().detach().numpy())
        # print(adj.shape)
        # print(x.shape)
        y = adj @ self.attend(x)
        return y


class GAT(nn.Module):
    def __init__(self, f_dim, h_dim, use_cuda=False):
        super(GAT, self).__init__()
        self.use_cuda = use_cuda

        self.map = nn.Linear(f_dim, h_dim)
        self.act = nn.Tanh()
        self.attentives = nn.ModuleList([AttentiveLayer(h_dim) for _ in range(2)])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.act(self.map(x))
        for attentive in self.attentives:
            x = attentive(x, adj)
        return x


class Classifier(nn.Module):
    def __init__(self, h_dim, l_dim, use_cuda=False):
        super(Classifier, self).__init__()
        self.use_cuda = use_cuda

        self.linear = nn.Linear(h_dim, l_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.softmax(self.linear(x))
        return l


if __name__ == '__main__':
    al = GAT(2, 4)
    x_ = torch.tensor([[1, 0], [0, 1], [0.5, 0.5]], dtype=torch.float32)
    adj_ = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    y_ = al(x_, adj_)
    print(y_.detach().numpy())
