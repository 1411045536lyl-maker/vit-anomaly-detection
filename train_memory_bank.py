import os
import argparse
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
        print("⏳ 正在加载官方 ImageNet 预训练权重...")
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.model.eval()

    def extract_spatial_features(self, x):
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.model.encoder.pos_embedding
        for layer in self.model.encoder.layers:
            x = layer(x)
        x = self.model.encoder.ln(x)
        patch_tokens = x[:, 1:]
        B, L, D = patch_tokens.shape
        grid_size = int(L ** 0.5)
        spatial_features = patch_tokens.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        return spatial_features

def parse_args():
    parser = argparse.ArgumentParser(description="构建 MVTec AD 正常样本的特征记忆库")
    parser.add_argument('--category', type=str, default='metal_nut', help='测试的物品类别名称 (如: metal_nut, bottle)')
    parser.add_argument('--data_root', type=str, default='./data/mvtec_ad', help='MVTec 数据集的根目录')
    parser.add_argument('--batch_size', type=int, default=16, help='提取特征时的批次大小')
    return parser.parse_args()

def build_memory_bank(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    print(f"📂 正在加载 {args.category} 的训练数据...")
    train_dataset = MVTecDataset(root_dir=args.data_root, category=args.category, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PretrainedViTExtractor().to(device)
    model.eval()

    memory_bank = []

    print("🧠 开始提取特征并构建记忆库...")
    with torch.no_grad():
        for images, labels, label_names, img_paths in tqdm(train_loader):
            images = images.to(device)
            features = model.extract_spatial_features(images)
            B, C, H, W = features.shape
            features_flatten = features.view(B, C, -1).transpose(1, 2).reshape(-1, C)
            memory_bank.append(features_flatten.cpu())

    memory_bank_tensor = torch.cat(memory_bank, dim=0)
    print(f"✅ 记忆库构建完成！")
    print(f"📊 共提取了 {memory_bank_tensor.shape[0]} 个 Patch 特征，特征维度为 {memory_bank_tensor.shape[1]}")

    save_dir = f'./weights/{args.category}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'memory_bank.pt')
    
    torch.save(memory_bank_tensor, save_path)
    print(f"💾 记忆库已成功保存至: {save_path}")

if __name__ == '__main__':
    args = parse_args()
    build_memory_bank(args)