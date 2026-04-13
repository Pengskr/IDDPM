import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, folder_a, folder_b, num_images, transform=None, threshold=0.9):
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
            img_a = self.transform(img_a)
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
        sample_a, sample_b, _ = dataset[random_idx]

        img_shape = sample_a.shape
        print(f"成功加载 {len(dataset)} 对图片，当前随机展示第 {random_idx} 对")
        print(f"单张图片张量尺寸 (C, H, W): {img_shape}")
        # 归一化回 0-255
        a_np = ((sample_a.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        b_np = ((sample_b.squeeze().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

        # 拼成一张横着的图
        combined = np.hstack([a_np, b_np])
        Image.fromarray(combined).save('../work_dirs/debug_view.png')
        print("训练数据示意图片 debug_view.png 已保存。")
    else:
        print("警告：数据集为空，请检查路径！")
    
    # 初始化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_P  = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
    dataloader_Mo = DataLoader(dataset, batch_size=25, shuffle=True, num_workers=4)

    return dataloader, dataloader_P, dataloader_Mo

def yield_dataloader(loader):
    while True:
        yield from loader