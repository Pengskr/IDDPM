import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, folder_a, folder_b, num_images, transform=None, threshold=0.99):
        self.dir_a = root_dir / folder_a
        self.dir_b = root_dir / folder_b
        self.transform = transform
        self.extension = ".jpg" 
        self.num_images = num_images
        self.threshold = threshold

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        img_name = f"{idx}{self.extension}"
        path_a = os.path.join(self.dir_a, img_name)
        path_b = os.path.join(self.dir_b, img_name)
        
        # 加载图片
        # .convert('RGB') 确保加载为3通道，如果是灰度图可改为 .convert('L')
        img_a = Image.open(path_a).convert('L')     # 0 是纯黑，255 是纯白
        img_b = Image.open(path_b).convert('L')

        # 应用预处理（如有）
        if self.transform:
            img_a = self.transform(img_a)   # ToTensor()会将L模式的PIL Image归一化
            img_b = self.transform(img_b)
        
        img_a = (img_a > self.threshold).float() * 2 - 1
        # img_b 反相：原本 > threshold 的部分变 -1，原本 <= threshold 的部分变 1
        img_b = (img_b <= self.threshold).float() * 2 - 1
        
        # 返回成对的张量
        return img_a, img_b, {}

def get_dataloader(root_folder_data, folder_Mo, folder_P, num_images, batch_size, image_size):
    data_transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
    ])
    dataset = PairedImageDataset(
        root_folder_data,
        folder_Mo,
        folder_P,
        num_images=num_images,
        transform=data_transform,   # Path有反相
        threshold=0.9
    )
    if len(dataset) > 0:
        random_idx = np.random.randint(len(dataset))
        sample_a, _, _ = dataset[random_idx]

        img_shape = sample_a.shape
        print(f"成功加载 {len(dataset)} 对图片,单张图片张量尺寸 (C, H, W): {img_shape}")
    else:
        print("警告：数据集为空，请检查路径！")
    
    # 初始化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader

def yield_dataloader(loader):
    while True:
        yield from loader

def show_dataloader(dataloader):
    sample_a, sample_b, _ = next(iter(dataloader))
    # 1. 提取第 1 对图像 (index 0)
    # 假设 sample_a 是 [batch_size, channels, height, width]
    img_a = sample_a[0].detach().cpu()
    img_b = sample_b[0].detach().cpu()

    # 2. 归一化并转换维度
    # 如果是单通道(灰度图)，squeeze 会去掉 C；如果是多通道，需要 transpose(1, 2, 0)
    def process_for_plot(tensor):
        # 归一化到 0-1 范围 (Matplotlib 对 float 类型的 0-1 支持更好)
        img = (tensor + 1) / 2
        img = img.clamp(0, 1).numpy()
        
        if img.shape[0] == 1: # 灰度图
            return img.squeeze(), 'gray'
        else: # RGB 
            return img.transpose(1, 2, 0), None

    data_a, cmap_a = process_for_plot(img_a)
    data_b, cmap_b = process_for_plot(img_b)

    # 3. 绘图
    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(data_a, cmap=cmap_a)
    plt.title("MAP")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(data_b, cmap=cmap_b)
    plt.title("PATH")
    plt.axis('off')

    plt.show()