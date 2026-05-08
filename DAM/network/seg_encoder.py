import torch
from torch import nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Modify from https://github.com/JWFangit/LOTVS-DADA/blob/master/SCAFNet/nets.py#L103
    Basic graph convolution layer (GCN) as in https://arxiv.org/abs/1609.02907
    Input: features=[batch, node, C_in], adj = [batch, node, node]
    Output: [batch, node, C_out]
    """
    def __init__(self, in_features, out_features, activation=None, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot uniform initialization for weights (類似 Keras 的 glorot_uniform)
        # nn.init.kaiming_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, features, basis):
        # features: (B, N, C_in)
        # basis:    (B, N, N)
        
        # 1. K.batch_dot(basis, features) -> torch.bmm(basis, features)
        # B x N x N @ B x N x C_in -> B x N x C_in
        supports = torch.bmm(basis, features)
        
        # 2. K.dot(supports, self.kernel) -> supports @ self.weight
        # B x N x C_in @ C_in x C_out -> B x N x C_out
        output = supports @ self.weight

        if self.bias is not None:
            output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
class SGcn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sim_embed1 = nn.Linear(in_channels, in_channels)
        self.sim_embed2 = nn.Linear(in_channels, in_channels)

        self.graph1 = GraphConvolution(in_channels, in_channels, activation=F.relu)
        self.graph2 = GraphConvolution(in_channels, in_channels, activation=F.relu)
        self.graph3 = GraphConvolution(in_channels, out_channels, activation=F.relu)

        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

    def get_adj(self, x):
        # x: (Batch, Nodes, Channels)
        sim1 = self.sim_embed1(x)
        sim2 = self.sim_embed2(x)

        # adj: (Batch, Nodes, Nodes)
        adj = torch.bmm(sim1, sim2.transpose(1, 2))
        scale = self.in_channels ** -0.5
        adj = adj * scale
        adj = F.softmax(adj, dim=-1)
        return adj

    def forward(self, x):
        # Input: (Batch, 3, H, W)
        b, c, h, w = x.size()
        x_reshaped = x.view(b, c, -1).permute(0, 2, 1)
        
        adj = self.get_adj(x_reshaped)
        
        outs = self.graph1(x_reshaped, adj)
        outs = self.ln1(outs)
        outs = self.graph2(outs, adj)
        outs = self.ln2(outs)
        outs = self.graph3(outs, adj)

        outs = torch.mean(outs, dim=1)
        outs = outs.view(b, self.out_channels, 1, 1)
        return outs

# SGCN Version
class Seg_encoder_v1(nn.Module):
    def __init__(self):
        super(Seg_encoder_v1, self).__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.sgcn = SGcn(in_channels=128, out_channels=128)
        self.final_norm = nn.BatchNorm3d(128)

    def forward(self, x):
        # Input: (Batch, 3, T, H, W)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        b, c, t, h, w = x.size()
        x = self.sgcn(x.permute(0, 2, 1, 3, 4).contiguous().view(b*t, c, h, w))
        x_out = self.final_norm(x.view(b, t, c ,1, 1).permute(0, 2, 1, 3, 4))
        return x_out

# GAP Version
class Seg_encoder_v2(nn.Module):
    def __init__(self):
        super(Seg_encoder_v2, self).__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.adapter = nn.Sequential(
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.Tanh()
        )

    def forward(self, x):
        # Input: (Batch, 3, T, H, W)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x) 
        x = x.mean(dim=(3, 4), keepdim=True)

        # x: (B, 128, T, 1, 1)
        b, c, t, _, _ = x.size()
        x = x.view(b, c, t)
        x = x.permute(0, 2, 1)
        x = self.adapter(x)

        x = x.permute(0, 2, 1)      # (B, 512, T)
        x = x.view(b, 512, t, 1, 1) # (B, 512, T, 1, 1)
        return x

# FCN version
class Seg_encoder_v3(nn.Module):
    def __init__(self, out_channels=256):
        super(Seg_encoder_v3, self).__init__()
        self.conv1a = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.conv1b = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.conv2b = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((1, 2, 2))

        self.adapter = nn.Sequential(
            nn.Conv3d(128, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input: (Batch, 3, Time, Height, Width)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.adapter(x)        
        # Output shape: (B, 256, T, H/4, W/4)
        return x