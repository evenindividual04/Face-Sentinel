import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from architectures.feature_extractor import FeatureExtractor
# ------- EfficientNetB4 with Attention, SE, and CBAM Blocks ------- #


# ------- SE Block (channel attention) ------- #
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ------- CBAM Block (channel + spatial attention) ------- #
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        scale = self.sigmoid_channel(avg_out + max_out)
        x = x * scale

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        scale_spatial = self.sigmoid_spatial(self.conv_spatial(pool))
        x = x * scale_spatial

        return x


# ------- Final Model ------- #
class EfficientNetB4AttSECBAMFusion(FeatureExtractor):
    def __init__(self):
        super(EfficientNetB4AttSECBAMFusion, self).__init__()

        self.base = EfficientNet.from_pretrained('efficientnet-b4')
        del self.base._fc  # remove default classifier

        # Constants from EfficientNet-B4
        self.mid_block_idx = 5       # For mid-level features
        self.att_block_idx = 9       # Spatial attention application point
        self.mid_channels = 32
        self.final_channels = 1792

        # Attention mechanisms
        self.attconv = nn.Conv2d(in_channels=self.mid_channels, out_channels=1, kernel_size=1)  # paper’s spatial attn
        self.cbam = CBAMBlock(self.final_channels)
        self.se = SEBlock(self.final_channels)

        # Feature fusion + classifier
        self.fused_dim = self.mid_channels + self.final_channels
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.fused_dim),
            nn.Linear(self.fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Stem
        x = self.base._swish(self.base._bn0(self.base._conv_stem(x)))

        mid_feat = None
        att_map = None

        # MBConv blocks
        for idx, block in enumerate(self.base._blocks):
            x = block(x)
            # Apply attention and extract mid-level features
            if idx == self.mid_block_idx:
                mid_feat = x  # B×160×14×14
                att_map = torch.sigmoid(self.attconv(mid_feat))  # B×1×14×14
                mid_feat_gap = F.adaptive_avg_pool2d(mid_feat, 1).squeeze(-1).squeeze(-1)  # B×160
            if idx == self.att_block_idx:
                # Resize att_map to match x's spatial size
                att_map_resized = F.interpolate(att_map, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = x * att_map_resized

        # Head
        x = self.base._swish(self.base._bn1(self.base._conv_head(x)))  # B×1792×7×7
        x = self.cbam(x)  # Apply CBAM (channel+spatial)
        x = self.se(x)    # Apply SE (channel)

        # Global pooling
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # B×1792

        # Mid + final fusion
        fused = torch.cat([mid_feat_gap, x], dim=1)  # B×(1792+160)

        # Classifier
        out = self.classifier(fused)
        return out

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
