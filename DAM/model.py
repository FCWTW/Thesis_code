import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from collections import OrderedDict
from video_swin_transformer import SwinTransformer3DBackbone
from einops import rearrange
import time
from utils import get_task_attribute_dict
import logging
import datetime

class DecoderSwin(nn.Module):
    def __init__(self, num_layers=4):
        super(DecoderSwin, self).__init__()
        
        self.upsampling = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(768, 384, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers == 1 else 3

        self.convtsp2 = nn.Sequential(
            nn.Conv3d(384, 192, kernel_size=(x, 3, 3), stride=(x, 1, 1), padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers < 4 else 5

        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False),
            nn.ReLU(),
            self.upsampling
        )

        x = 1 if num_layers < 3 else 5
        layers = [('conv3_1', nn.Conv3d(96, 64, kernel_size=(x,3,3), stride=(x,1,1), padding=(0,1,1), bias=False)),
                              ('relu_1', nn.ReLU()),
                              ('up_1', self.upsampling),
                              ('conv3_2', nn.Conv3d(64, 32, kernel_size=(1,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)),
                              ('relu_2', nn.ReLU()),
                              ('up_2', self.upsampling)
                              
                ]
        if num_layers == 1:
            layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(2,1,1), bias=True)))
        else:
            layers.append(('conv3_3', nn.Conv3d(32, 1, kernel_size=(1,1,1), stride=(1,1,1), bias=True)))

        layers.append(('sigm', nn.Sigmoid()))

        self.convtsp4 = nn.Sequential(OrderedDict(layers))

    def forward(self, y):
        if not isinstance(y, list):
            raise ValueError(f'ERROR: input to decoder should be a list!')

        if len(y) >= 1:
            z = self.convtsp1(y[0])

        if len(y) >= 2:
            z = torch.cat((z,y[1]), 2)
        
        z = self.convtsp2(z)

        if len(y) >= 3:
            z = torch.cat((z,y[2]), 2)
        
        z = self.convtsp3(z)

        if len(y) == 4:
            z = torch.cat((z,y[3]), 2)
        
        z = self.convtsp4(z)
        
        z = z.view(z.size(0), z.size(3), z.size(4))
        return z

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
    
# SGCN Seg encoder + Cross-Attention
class DAM_v1(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM_v1, self).__init__()
        self.attn_logger = logging.getLogger('Attention Weight')
        self.attn_logger.setLevel(logging.INFO)
        if not self.attn_logger.handlers:
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            fh = logging.FileHandler(f'cache/{filename}', mode='w')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fh.setFormatter(formatter)
            self.attn_logger.addHandler(fh)
        self.log_step_counter = 0
        self.log_frequency = 100

        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
                                                train_backbone=train_backbone)

        self.num_encoder_layers = num_encoder_layers

        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}, \
                            should be between 1 and 4.')

        self.seg_encoder = Seg_encoder_v1()
        self.add_and_norm = transformer_params['add_and_norm']
        self.fuse_idx = transformer_params['fuse_idx']
        self.num_att_heads = transformer_params['num_att_heads']

        self.multihead_attn = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.seg_projectors = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]

        for i in range(4):
            if i in self.fuse_idx:
                self.multihead_attn.append(
                    nn.MultiheadAttention(embed_dim=embed_dims[i],
                                          num_heads=self.num_att_heads[i], 
                                          bias=True)
                )
                self.norm.append(nn.LayerNorm(embed_dims[i]))
                self.seg_projectors.append(nn.Linear(512, embed_dims[i]))
            else:
                self.multihead_attn.append(nn.Identity())
                self.norm.append(nn.Identity())
                self.seg_projectors.append(nn.Identity())
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, seg_input):
        # b_out: (B, C, T, H, W)
        b_out = self.backbone_3d(x)
        b_s = [b.shape for b in b_out]

        # seg_out: (B, 512, T, 1, 1)
        seg_out = self.seg_encoder(seg_input)
        if self.training:
            self.log_step_counter += 1

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:    
                # Query
                b_flat = b.flatten(2).permute((2, 0, 1))

                # Key and Value
                curr_seg = seg_out.squeeze(-1).squeeze(-1)      # (B, 512, T)
                curr_seg = curr_seg.permute(0, 2, 1)            # (B, T, 512) for Linear
                curr_seg = self.seg_projectors[idx](curr_seg)   # (B, T, C)
                seg_flat = curr_seg.permute(1, 0, 2)            # (T, B, C)

                fused_out, attn_weights = self.multihead_attn[idx](
                    b_flat, seg_flat, seg_flat, need_weights=True
                )
                
                # --- Monitoring code for attention weight ---
                if self.training and (self.log_step_counter % self.log_frequency == 0 or self.log_step_counter == 1):
                    with torch.no_grad():
                        max_w = attn_weights.max().item()
                        mean_w = attn_weights.mean().item()
                        std_w = attn_weights.std(dim=-1).mean().item()
                        log_msg = f"[Step {self.log_step_counter}][Layer {idx}] Attn Stats -> Max: {max_w:.4f}, Mean: {mean_w:.4f}, Std: {std_w:.6f}"
                        if torch.isnan(attn_weights).any():
                            self.attn_logger.warning(f"[Step {self.log_step_counter}][Layer {idx}] ALERT: Attention weights contains NaN!")
                        else:
                            self.attn_logger.info(log_msg)
                
                fused_out = self.dropout(fused_out)
                if self.add_and_norm:
                    fused_out += b_flat
                    fused_out = self.norm[idx](fused_out)

                fused_out = fused_out.permute((1, 2, 0))
                b_out[idx] = fused_out.view(*b_s[idx])
        return self.decoder(b_out[:self.num_encoder_layers])
    
# FCN Seg encoder + Concat
class DAM_2(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM_2, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256
        self.seg_encoder = Seg_encoder_v3(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # Number of channels after concatenation = Backbone feature dimension + Seg feature dimension (256)
                in_channels = embed_dims[i] + self.seg_channels
                out_channels = embed_dims[i]

                # Late Fusion Adapter
                fusion_layer = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
                nn.init.xavier_uniform_(fusion_layer[0].weight)
                self.fusion_convs.append(fusion_layer)
            else:
                self.fusion_convs.append(nn.Identity())

    def forward(self, x, seg_input):
        with torch.no_grad():
            b_out = self.backbone_3d(x)

        # Output shape: (B, 256, T_seg, H_seg, W_seg)
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b: (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                fused = torch.cat([b, aligned_seg], dim=1)
                fused = self.fusion_convs[idx](fused)
                b_out[idx] = self.dropout(fused)
        return self.decoder(b_out[:self.num_encoder_layers])

if __name__ == "__main__":
    batch_size = 4
    time_steps = 16
    height = 128
    width = 128

    # (Batch size, channel, time step, height, width)
    x_dummy = torch.randn(batch_size, 3, time_steps, height, width)
    y_dummy = torch.randn(batch_size, 3, time_steps, height, width)

    model = DAM_2()
    if torch.cuda.is_available():
        model = model.cuda()
        x_dummy = x_dummy.cuda()
        y_dummy = y_dummy.cuda()
    
    out = model(x_dummy, y_dummy)
    print(f"Input 1 shape: {x_dummy.shape}")
    print(f"Input 2 shape: {y_dummy.shape}")
    print(f"Output shape: {out.shape}")