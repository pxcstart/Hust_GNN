from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import global_add_pool

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Readout(nn.Module):
    def __init__(self, emb_size):
        super(Readout, self).__init__()
        self.hidden_dim = 200 # the out_channel from GatedGraphConv
        self.emb_size = emb_size
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim+emb_size,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
        )
        self.out_fn = nn.Linear(64,1)
        self.act_fn = nn.Sigmoid()

    def forward(self, h, x, batch):
        feat = torch.concat([h,x],dim=-1)
        feat = self.mlp(feat)
        global_feat = global_add_pool(feat,batch)
        global_feat = self.act_fn(self.out_fn(global_feat).squeeze())
        return global_feat
    
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):
    def __init__(
        self, max_nodes, hidden_size, embed_size
    ):
        super(Conv, self).__init__()
        self.max_nodes = max_nodes
        self.conv_y1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3)
        self.pool_y1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_y2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
        self.pool_y2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_y = nn.Linear(hidden_size, 1)

        self.conv_z1 = nn.Conv1d(in_channels=hidden_size + embed_size, out_channels=hidden_size + embed_size, kernel_size=3)
        self.pool_z1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv_z2 = nn.Conv1d(in_channels=hidden_size + embed_size, out_channels=hidden_size + embed_size, kernel_size=1)
        self.pool_z2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc_z = nn.Linear(hidden_size + embed_size, 1)

        init_weights(self.conv_y1)
        init_weights(self.conv_y2)
        init_weights(self.conv_z1)
        init_weights(self.conv_z2)
        init_weights(self.fc_y)
        init_weights(self.fc_z)

        self.sigmoid = nn.Sigmoid()

    def forward(self, h, x):
        h = h.reshape(-1, self.max_nodes, h.shape[-1])
        x = x.reshape(-1, self.max_nodes, x.shape[-1])
        combined = torch.cat((h, x), dim=-1)
        y1 = self.pool_y1(F.relu(self.conv_y1(h.transpose(-2, -1))))
        y2 = self.pool_y2(F.relu(self.conv_y2(y1))).transpose(-2, -1)
        y = self.fc_y(y2)

        z1 = self.pool_z1(F.relu(self.conv_z1(combined.transpose(-2, -1))))
        z2 = self.pool_z2(F.relu(self.conv_z2(z1))).transpose(-2, -1)
        z = self.fc_z(z2)

        mult = torch.mul(y, z)
        avg = mult.mean(dim=1).squeeze(-1)
        return self.sigmoid(avg)

class GatedGraphRecurrentLayer(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super(GatedGraphRecurrentLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv_ast = GCNConv(hidden_size, hidden_size)
        self.conv_cfg = GCNConv(hidden_size, hidden_size)
        self.conv_dfg = GCNConv(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
    
    def agg(self, a_ast, a_cfg, a_dfg):
        return a_ast + a_cfg + a_dfg

    def forward(self, x, edge_ast, edge_cfg, edge_dfg):
        h = x
        for _ in range(self.num_layers):
            a_ast = self.conv_ast(h, edge_ast)
            a_cfg = self.conv_cfg(h, edge_cfg)
            a_dfg = self.conv_dfg(h, edge_dfg)
            out, h = self.gru(self.agg(a_ast, a_cfg, a_dfg).unsqueeze(1), h.unsqueeze(0))
            h = h[-1, :, :]
        return h
    
class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        # self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device) 
        self.ggc = GatedGraphRecurrentLayer(gated_graph_conv_args["out_channels"], 6).to(device)
        self.linear2 = nn.Linear(101, emb_size).to(device)
        self.linear1 = nn.Linear(101, gated_graph_conv_args["out_channels"]).to(device)
        init_weights(self.linear1)
        init_weights(self.linear2)
        self.emb_size=emb_size
        hidden_size = gated_graph_conv_args["out_channels"]
        self.readout = Conv(max_nodes, hidden_size, emb_size).to(device)  

    def forward(self, data):
        # breakpoint()
        x, edge_index_ast, edge_index_cfg, edge_index_dfg = data[0].x, data[0].edge_index , data[1].edge_index, data[2].edge_index
        x1 = F.relu(self.linear1(x))
        x1 = self.ggc(x1, edge_index_ast, edge_index_cfg, edge_index_dfg)
        x2 = F.relu(self.linear2(x))
        x = self.readout(x1, x2)
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

