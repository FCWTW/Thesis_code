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

class Seg_encoder(nn.Module):
    def __init__(self, out_channels=256):
        super(Seg_encoder, self).__init__()
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

        # 【修改 1】移除 Linear，改用 1x1x1 Conv3d 來保留空間維度 (Fully Convolutional)
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
        
        # 【修改 2】徹底移除 x.mean(dim=(3, 4)) GAP 操作
        # 目前 x 的形狀是 (B, 128, T, H/4, W/4)

        # 【修改 3】直接通過 3D 卷積進行通道轉換
        x = self.adapter(x)       
        
        # Output shape: (B, 256, T, H/4, W/4)
        return x
    
class DAM(nn.Module):
    def __init__(self, 
                  num_encoder_layers=4,
                  train_backbone=False,
                  pretrained_backbone=False,
                  transformer_params=None,
                  **kwargs
                ):

        super(DAM, self).__init__()
        self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone, train_backbone=train_backbone)
        # self.backbone_3d = SwinTransformer3DBackbone(pretrained=pretrained_backbone,
        #                                         train_backbone=train_backbone,
        #                                         drop_path_rate=0,
        #                                         attn_drop_rate=0,
        #                                         drop_rate=0)
        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers in [1, 2, 3, 4]:
            self.decoder = DecoderSwin(num_encoder_layers)
        else:
            raise ValueError(f'ERROR: unsupported num_encoder_layers={num_encoder_layers}')

        self.seg_channels = 256
        self.seg_encoder = Seg_encoder(out_channels=self.seg_channels)
        self.fuse_idx = transformer_params['fuse_idx']

        self.fusion_convs = nn.ModuleList()
        embed_dims = [768, 384, 192, 96]
        self.dropout = nn.Dropout(0.5)

        for i in range(4):
            if i in self.fuse_idx:
                # 拼接後的輸入通道數 = Backbone特徵維度 + Seg特徵維度(256)
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

        # Output shape: (B, 256, T_seg, H_seg, W_seg) -> 保留了完整的空間幾何資訊
        seg_out = self.seg_encoder(seg_input)

        for idx, b in enumerate(b_out):
            if idx in self.fuse_idx:
                # b 的形狀為 (B, C, T_b, H, W)
                B, C, T_b, H, W = b.shape

                # --- 【修改 5】動態時空對齊 (Spatio-Temporal Alignment) ---
                # 不再使用暴力擴展 (expand)，而是使用 trilinear 插值
                # 直接將 Seg 特徵縮放對齊到當前 Backbone 層的時間與空間解析度
                aligned_seg = F.interpolate(
                    seg_out, 
                    size=(T_b, H, W), 
                    mode='trilinear', 
                    align_corners=False
                )

                # --- 通道拼接 (Concatenation) ---
                # 現在兩者的 T, H, W 都「真實」匹配了！左上角的車會對準左上角的特徵
                # fused 形狀變為 (B, C + 256, T_b, H, W)
                fused = torch.cat([b, aligned_seg], dim=1)
                
                # 特徵融合與降維
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

    model = SCOUT_seg_v2()
    if torch.cuda.is_available():
        model = model.cuda()
        x_dummy = x_dummy.cuda()
        y_dummy = y_dummy.cuda()
    
    out = model(x_dummy, y_dummy)
    print(f"Input 1 shape: {x_dummy.shape}")
    print(f"Input 2 shape: {y_dummy.shape}")
    print(f"Output shape: {out.shape}")