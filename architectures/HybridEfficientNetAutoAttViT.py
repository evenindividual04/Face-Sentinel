import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=32, in_channels=3, embed_dim=512):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=2, heads=8):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class HybridEfficientNetAutoAttViT(nn.Module):
    def __init__(self, transformer_dim=512, patch_size=32, vit_depth=2, heads=8):
        super(HybridEfficientNetAutoAttViT, self).__init__()

        # EfficientNet B4 with paper's attention
        self.cnn = EfficientNet.from_pretrained('efficientnet-b4')
        self.cnn._fc = nn.Identity()  # Remove final FC

        self.cnn_feat_dim = self.cnn._conv_head.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Linear projection to transformer dim
        self.cnn_proj = nn.Linear(self.cnn_feat_dim, transformer_dim)

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size=224, patch_size=patch_size, in_channels=3, embed_dim=transformer_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))

        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim=transformer_dim, depth=vit_depth, heads=heads)

        # Final classification head
        self.classifier = nn.Linear(transformer_dim, 1)

    def forward(self, x):
        B = x.shape[0]

        # CNN feature
        cnn_feat = self.cnn.extract_features(x)  # [B, C, H, W]
        cnn_feat = self.pool(cnn_feat).view(B, -1)  # [B, C]
        cnn_token = self.cnn_proj(cnn_feat).unsqueeze(1)  # [B, 1, D]

        # Patch tokens
        patch_tokens = self.patch_embed(x)  # [B, N, D]

        # Combine [CLS] + CNN token + patches
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        tokens = torch.cat([cls_token, cnn_token, patch_tokens], dim=1)  # [B, 1+1+N, D]

        # Transformer
        encoded = self.transformer(tokens)

        # Classification from CLS token
        cls_output = encoded[:, 0]  # [B, D]
        out = self.classifier(cls_output)  # [B, 1]
        return out
    
    def get_trainable_parameters(self):
        return self.parameters()
    
    @staticmethod
    def get_normalizer():
        from torchvision import transforms
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


