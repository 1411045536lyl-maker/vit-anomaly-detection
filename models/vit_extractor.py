from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

IntPair = Union[int, Tuple[int, int]]

def _pair(value: IntPair) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return value, value

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_h, img_w = _pair(img_size)
        patch_h, patch_w = _pair(patch_size)
        if img_h % patch_h != 0 or img_w % patch_w != 0:
            raise ValueError("img_size must be divisible by patch_size.")
        self.grid_size = (img_h // patch_h, img_w // patch_w)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class ConvStemPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 768,
        stem_channels: Sequence[int] = (64, 128),
    ) -> None:
        super().__init__()
        img_h, img_w = _pair(img_size)
        layers: List[nn.Module] = []
        current_channels = in_chans
        stride_total = 1

        for channels in stem_channels:
            layers.extend(
                [
                    nn.Conv2d(current_channels, channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                ]
            )
            current_channels = channels
            stride_total *= 2

        if img_h % stride_total != 0 or img_w % stride_total != 0:
            raise ValueError("img_size must be divisible by the conv stem total stride.")

        layers.append(nn.Conv2d(current_channels, embed_dim, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*layers)
        self.grid_size = (img_h // stride_total, img_w // stride_total)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: Tensor, return_attention: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size, num_tokens, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_map = attn if return_attention else None
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map

class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attention_dropout,
            proj_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout=dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: Tensor, return_attention: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        attn_out, attn_map = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_map

def build_2d_sincos_position_embedding(
    embed_dim: int,
    grid_size: Tuple[int, int],
    with_cls_token: bool,
) -> Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("Sinusoidal position embedding requires embed_dim % 4 == 0.")
    grid_h = torch.arange(grid_size[0], dtype=torch.float32)
    grid_w = torch.arange(grid_size[1], dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size[0], grid_size[1])

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (10000 ** omega)

    out_h = torch.einsum("m,d->md", grid[0].reshape(-1), omega)
    out_w = torch.einsum("m,d->md", grid[1].reshape(-1), omega)
    pos_embed = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w)],
        dim=1,
    )
    if with_cls_token:
        cls_token = torch.zeros(1, embed_dim, dtype=torch.float32)
        pos_embed = torch.cat([cls_token, pos_embed], dim=0)
    return pos_embed.unsqueeze(0)

@dataclass
class ViTConfig:
    img_size: int = 224      # 默认修改为工业图像常用尺寸
    patch_size: int = 16     # 默认修改为 16
    in_chans: int = 3
    num_classes: int = 10    # 虽然保留了参数，但模型中已不再使用
    embed_dim: int = 768     # 默认修改为标准 ViT-Base 维度
    depth: int = 12          # 默认修改为标准 ViT-Base 深度
    num_heads: int = 12      # 默认修改为标准 ViT-Base 头数
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0
    qkv_bias: bool = True
    use_cls_token: bool = True
    pooling: str = "cls"
    pos_embed_type: str = "learnable"
    model_type: str = "pure_vit"
    conv_stem_channels: Tuple[int, ...] = (64, 128)

class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        
        # 实例化 Patch 嵌入层
        if config.model_type == "pure_vit":
            self.patch_embed = PatchEmbed(
                img_size=config.img_size,
                patch_size=config.patch_size,
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
            )
        else:
            self.patch_embed = ConvStemPatchEmbed(
                img_size=config.img_size,
                in_chans=config.in_chans,
                embed_dim=config.embed_dim,
                stem_channels=config.conv_stem_channels,
            )
            
        self.num_patches = self.patch_embed.num_patches
        self.use_cls_token = config.use_cls_token

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        else:
            self.register_parameter("cls_token", None)

        token_count = self.num_patches + int(self.use_cls_token)
        
        if config.pos_embed_type == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, token_count, config.embed_dim))
        elif config.pos_embed_type == "sinusoidal":
            pos_embed = build_2d_sincos_position_embedding(
                config.embed_dim,
                self.patch_embed.grid_size,
                self.use_cls_token,
            )
            self.register_buffer("pos_embed", pos_embed, persistent=False)
        else:
            self.register_buffer("pos_embed", torch.zeros(1, token_count, config.embed_dim), persistent=False)

        self.pos_drop = nn.Dropout(config.dropout)
        drop_path_values = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        
        self.blocks = nn.ModuleList([
            EncoderBlock(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                drop_path=drop_path_values[i],
            )
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # ==========================================
        # 修改点 1：删除了 self.head 分类头
        # ==========================================

        self._init_weights()

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor, return_attention: bool = False) -> Tuple[Tensor, List[Tensor]]:
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.pos_drop(x)

        attention_maps: List[Tensor] = []
        for block in self.blocks:
            x, attn_map = block(x, return_attention=return_attention)
            if attn_map is not None:
                attention_maps.append(attn_map.detach())
                
        x = self.norm(x)
        return x, attention_maps

    def forward(self, x: Tensor) -> Tensor:
        # ==========================================
        # 修改点 2：不再进行池化和分类，直接返回所有特征序列
        # 输出形状： [Batch_size, Num_tokens, Embed_dim]
        # ==========================================
        tokens, _ = self.forward_features(x, return_attention=False)
        return tokens

    def extract_spatial_features(self, x: Tensor) -> Tensor:
        """
        ==========================================
        修改点 3：专门为 MVTec 异常检测设计的特征提取接口
        将一维的 Token 序列重构回二维图像特征图 (Feature Map)
        ==========================================
        """
        tokens, _ = self.forward_features(x, return_attention=False)
        
        # 丢弃 CLS token，只保留具有空间意义的 Patch tokens
        if self.use_cls_token:
            patch_tokens = tokens[:, 1:]
        else:
            patch_tokens = tokens
            
        B, L, D = patch_tokens.shape
        grid_h, grid_w = self.patch_embed.grid_size
        
        # 将形状从 [B, H*W, C] 转换为 CNN 标准的 [B, C, H, W]
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, D, grid_h, grid_w)
        return spatial_features

    @torch.no_grad()
    def get_last_attention_map(self, x: Tensor) -> Tensor:
        _, attention_maps = self.forward_features(x, return_attention=True)
        if not attention_maps:
            raise RuntimeError("Attention maps were not produced.")
        return attention_maps[-1]