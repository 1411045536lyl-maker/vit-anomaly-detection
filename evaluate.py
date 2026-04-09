import os
import cv2
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 导入数据加载器
from utils.dataset import MVTecDataset

class PretrainedViTExtractor(torch.nn.Module):
    """封装官方预训练 ViT (与 train_memory_bank 保持完全一致)"""
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
    parser = argparse.ArgumentParser(description="执行 MVTec AD 的异常检测与热力图生成")
    parser.add_argument('--category', type=str, default='metal_nut', help='测试的物品类别名称')
    parser.add_argument('--data_root', type=str, default='./data/mvtec_ad', help='MVTec 数据集的根目录')
    return parser.parse_args()

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在使用设备: {device}")

    memory_bank_path = f'./weights/{args.category}/memory_bank.pt'
    if not os.path.exists(memory_bank_path):
        print(f"❌ 找不到 {args.category} 的记忆库，请先运行: python train_memory_bank.py --category {args.category}")
        return
    memory_bank = torch.load(memory_bank_path).to(device)
    print(f"🧠 成功加载记忆库，特征总数: {memory_bank.shape[0]}")

    model = PretrainedViTExtractor().to(device)
    model.eval()

    print(f"📂 正在加载 {args.category} 的测试数据...")
    test_dataset = MVTecDataset(root_dir=args.data_root, category=args.category, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    save_dir = f'./outputs/{args.category}/heatmaps'
    os.makedirs(save_dir, exist_ok=True)

    all_image_scores = []  
    all_image_labels = []  

    print("🔍 开始进行异常检测并生成热力图...")
    with torch.no_grad():
        for idx, (image, label, label_names, img_paths) in enumerate(test_loader):
            image = image.to(device)
            img_path = img_paths[0]
            label_name = label_names[0]
            true_label = label.item() 

            features = model.extract_spatial_features(image)
            B, C, H, W = features.shape
            query_features = features.view(C, -1).transpose(0, 1)

            distances = torch.cdist(query_features, memory_bank)
            min_distances, _ = torch.min(distances, dim=1)
            
            image_anomaly_score = min_distances.max().item()
            all_image_scores.append(image_anomaly_score)
            all_image_labels.append(true_label)

            anomaly_map = min_distances.view(1, 1, H, W)
            anomaly_map = F.interpolate(anomaly_map, size=(224, 224), mode='bilinear', align_corners=False)
            anomaly_map = anomaly_map.squeeze().cpu().numpy()

            anomaly_map_norm = anomaly_map - anomaly_map.min()
            if anomaly_map_norm.max() != 0:
                anomaly_map_norm = anomaly_map_norm / anomaly_map_norm.max()
            anomaly_map_norm = (anomaly_map_norm * 255).astype(np.uint8)

            heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
            img_unnorm = image[0] * std + mean
            img_unnorm = img_unnorm.permute(1, 2, 0).cpu().numpy()
            img_unnorm = np.clip(img_unnorm, 0, 1)
            img_unnorm = (img_unnorm * 255).astype(np.uint8)
            
            img_unnorm_bgr = cv2.cvtColor(img_unnorm, cv2.COLOR_RGB2BGR)

            overlay = cv2.addWeighted(img_unnorm_bgr, 0.5, heatmap, 0.5, 0)
            combined_img = np.hstack((img_unnorm_bgr, overlay))

            img_name = os.path.basename(img_path)
            save_file = os.path.join(save_dir, f"{label_name}_{img_name}")
            cv2.imwrite(save_file, combined_img)

    print("\n" + "="*50)
    print("📈 全局性能评估 (Global Evaluation)")
    print("="*50)
    try:
        image_auroc = roc_auc_score(all_image_labels, all_image_scores)
        print(f"✨ Image-level AUROC: {image_auroc * 100:.2f}%")
    except ValueError:
        print("⚠️ 计算 AUROC 失败。")
    print("="*50)

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)