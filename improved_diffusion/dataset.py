import os
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch as th
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
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
        
        # 加载图片：除了PyTorch (Tensor) 是CxHxW，其余(PIL,Numpy,OpenCV,Matplotlib)几乎都是HxWxC
        img_a = Image.open(path_a).convert('L')     # 灰度图 M_o
        img_b = Image.open(path_b).convert('L')     # 灰度图 P
        img_c = Image.open(path_a).convert('RGB')   # 彩色地图 M_r (包含红蓝标记)

        # 预处理
        if self.transform:
            img_a = self.transform(img_a)   # ToTensor()不仅把像素值从 [0, 255] 归一化到 [0, 1]，还会自动把维度从 (H, W, C) 调换成 (C, H, W)
            img_b = self.transform(img_b)
            img_c = self.transform(img_c)
        
        img_a = (img_a > self.threshold).float() * 2 - 1
        img_b = (img_b <= self.threshold).float() * 2 - 1   # img_b 反相：原本 > threshold 的部分变 -1，原本 <= threshold 的部分变 1
        img_c = img_c * 2.0 - 1.0
        
        # 返回成对的张量
        return img_a, img_c, img_b, {}, img_name    # M_o, M_r, P, cond, img_name

def get_dataloader(root_folder_data, folder_Mo, folder_P, num_images, batch_size, image_size, shuffle = True):
    data_transform = Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
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
        sample_Mo, sample_Mr, sample_P, _, _ = dataset[random_idx]

        print(f"成功加载 {len(dataset)} 对图片,M_o的尺寸为 (C, H, W): {sample_Mo.shape};M_r的尺寸为 (C, H, W): {sample_Mr.shape};P的尺寸为 (C, H, W): {sample_P.shape}")
    else:
        print("警告：数据集为空，请检查路径！")
    
    # 初始化 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

def yield_dataloader(loader):
    while True:
        yield from loader

def show_dataloader(dataloader):
    sample_Mo, sample_Mr, sample_P, _, _ = next(iter(dataloader))
    # 1. 提取第 1 对图像 (index 0)
    # 假设 sample_a 是 [batch_size, channels, height, width]
    img_Mo = sample_Mo[0].detach().cpu()
    img_Mr = sample_Mr[0].detach().cpu()
    img_P  = sample_P[0].detach().cpu()

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

    data_Mo, cmap_Mo = process_for_plot(img_Mo)
    data_Mr, cmap_Mr = process_for_plot(img_Mr)
    data_P, cmap_P   = process_for_plot(img_P)

    # 3. 绘图
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(data_Mo, cmap=cmap_Mo)
    plt.title("Mo")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(data_Mr, cmap=cmap_Mr)
    plt.title("Mr")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(data_P, cmap=cmap_P)
    plt.title("P")
    plt.axis('off')

    plt.show()

def show_samples(args, arr, M_o, P, img_name):
    n_display = args.num_samples
    M_o_disp = M_o[:n_display].detach().cpu()
    P_disp = P[:n_display].detach().cpu()

    # 将采样得到的 numpy 数组转回 tensor 并还原维度顺序为 [N, C, H, W]
    imgs_disp = th.from_numpy(arr[:n_display]).permute(0, 3, 1, 2).float()

    # 归一化到 [0, 255]
    # M_o 和 P 原始通常在 [-1, 1]，imgs_disp 如果已经是 uint8 转回来的，就在 [0, 255]
    M_o_disp = (M_o_disp + 1) / 2 * 255
    P_disp = (P_disp + 1) / 2 * 255
    # 如果 imgs_disp 已经是 0-255 范围，则不需要再处理，否则按需归一化

    # 统一转换为 3 通道
    M_o_disp = ensure_rgb(M_o_disp)
    P_disp = ensure_rgb(P_disp)
    imgs_disp = ensure_rgb(imgs_disp)

    # 给每张图加框线 (Padding) 
    pad_width = 1  # 线条宽度（像素）
    pad_value = 127 # 线条颜色：0为黑，255为白，127为灰色

    # F.pad 参数顺序是 (左, 右, 上, 下)
    # 我们给右边和下边加 pad，这样拼接后就有分割线了
    def add_border(x):
        return th.nn.functional.pad(x, (0, pad_width, 0, pad_width), value=pad_value)

    M_o_disp = add_border(M_o_disp)
    P_disp = add_border(P_disp)
    imgs_disp = add_border(imgs_disp)

    # 堆叠成三元组 [N, 3, 1, H, W]
    # 顺序：M_o (地图), P (真实路径), imgs (生成路径)
    combined = th.stack([M_o_disp, P_disp, imgs_disp], dim=1)

    # 使用 einops 排布
    # 设置每行显示的样本组数 (每组包含 3 张图)
    n_groups_per_row = 4  # 每行显示 4 组路径对比
    b1 = n_display // n_groups_per_row
    b2 = n_groups_per_row

    # 排布逻辑：纵向堆叠样本 (b1*h)，横向排布 (b2组 * 每组3张 * 宽度w)
    res = einops.rearrange(combined, 
                        '(b1 b2) p c h w -> (b1 h) (b2 p w) c', 
                        b1=b1, b2=b2, p=3)

    # 转换类型用于显示
    res_np = res.clamp(0, 255).numpy().astype(np.uint8)

    # 直接在 Notebook 中显示
    plt.figure(figsize=(20, 10)) # 增加宽度以适应 1:3 的比例
    plt.imshow(res_np)
    plt.axis('off')
    plt.title("Comparison: Map (Mo/Mr) | Ground Truth (P) | Sampled Path")
    plt.show()

    print(img_name)

def ensure_rgb(x):
        # x shape: [B, C, H, W]
        if x.shape[1] == 1:
            # 将 [B, 1, H, W] 广播/复制为 [B, 3, H, W]
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            return x
        else:
            # 处理特殊情况（如4通道），取前3通道
            return x[:, :3, :, :]