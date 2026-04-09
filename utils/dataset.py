import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, is_train=True, img_size=224):
        """
        MVTec AD 数据集加载器
        :param root_dir: MVTec 数据集的根目录 (例如: './data')
        :param category: 具体的类别名称 (例如: 'metal_nut')
        :param is_train: True 表示训练模式(只读取正常图片)，False 表示测试模式(读取所有图片)
        :param img_size: ViT 默认接受 224x224 的尺寸
        """
        self.is_train = is_train
        self.category_dir = os.path.join(root_dir, category)
        
        # 定义 ViT 的标准图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # 使用 ImageNet 的标准均值和方差进行归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_paths = []
        self.labels = []      # 0 代表正常(good)，1 代表异常(defect)
        self.label_names = [] # 记录具体的缺陷类型(如 scratch, crack)

        self._load_data()

    def _load_data(self):
        phase = 'train' if self.is_train else 'test'
        data_dir = os.path.join(self.category_dir, phase)

        # 增加一个容错判断，防止路径写错
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"找不到数据目录: {data_dir}，请确认数据集是否已正确解压到该位置。")

        # 遍历该文件夹下的所有子文件夹 (good, scratch, crack 等)
        for img_type in os.listdir(data_dir):
            type_dir = os.path.join(data_dir, img_type)
            if not os.path.isdir(type_dir):
                continue

            # 获取所有 .png 图片
            img_files = glob(os.path.join(type_dir, '*.png'))
            
            for img_file in img_files:
                self.image_paths.append(img_file)
                self.label_names.append(img_type)
                # 'good' 类别标签为 0，其他各种缺陷标签为 1
                self.labels.append(0 if img_type == 'good' else 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        label_name = self.label_names[idx]

        return image, label, label_name, img_path