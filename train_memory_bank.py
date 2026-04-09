import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 导入数据加载器
from utils.dataset import MVTecDataset

class PretrainedViTExtractor(torch.nn.Module):
    """封装官方预训练 ViT，使其输出适配我们业务逻辑的二维空间特征"""
    def __init__(self):
        super().__init__()
        # 第一次运行会自动下载 ImageNet 权重 (约 300MB)
        print("⏳ 正在加载官方 ImageNet 预训练权重...")
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.eval()

    def extract_spatial_features(self, x):
        # 1. 图像预处理与 Patch 嵌入
        x = self.model._process_input(x)
        n = x.shape[0]

        # 2. 拼接 CLS token
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # 3. 注入位置编码
        x = x + self.model.encoder.pos_embedding

        # 4. 前向传播经过所有 Transformer 层
        for layer in self.model.encoder.layers:
            x = layer(x)
        x = self.model.encoder.ln(x)

        # 5. 丢弃 CLS token，仅保留 196 个 Patch 的特征
        patch_tokens = x[:, 1:] # 形状: [B, 196, 768]
        
        # 6. 将一维序列重新折叠为二维特征图
        B, L, D = patch_tokens.shape
        grid_size = int(L ** 0.5) # 196 -> 14x14
        
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return spatial_features

def build_memory_bank(category='metal_nut', data_root='./data/mvtec_ad'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    print(f"📂 正在加载 {category} 的训练数据...")
    train_dataset = MVTecDataset(root_dir=data_root, category=category, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ==========================================
    # 使用拥有先验知识的预训练特征提取器
    # ==========================================
    model = PretrainedViTExtractor().to(device)
    model.eval()

    memory_bank = []

    print("🧠 开始提取特征并构建记忆库...")
    with torch.no_grad():
        for images, labels, label_names, img_paths in tqdm(train_loader):
            images = images.to(device)
            
            # [B, 768, 14, 14]
            features = model.extract_spatial_features(images)
            
            B, C, H, W = features.shape
            features_flatten = features.view(B, C, -1).transpose(1, 2).reshape(-1, C)
            
            memory_bank.append(features_flatten.cpu())

    memory_bank_tensor = torch.cat(memory_bank, dim=0)
    print(f"✅ 记忆库构建完成！")
    print(f"📊 共提取了 {memory_bank_tensor.shape[0]} 个 Patch 特征，特征维度为 {memory_bank_tensor.shape[1]}")

    save_dir = f'./weights/{category}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'memory_bank.pt')
    
    torch.save(memory_bank_tensor, save_path)
    print(f"💾 记忆库已成功保存至: {save_path}")

if __name__ == '__main__':
    build_memory_bank(category='metal_nut', data_root='./data/mvtec_ad')