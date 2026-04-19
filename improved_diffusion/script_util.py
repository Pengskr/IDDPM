import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, SpacedDiffusion_without_MFF_MCA, space_timesteps
from .unet import SuperResModel, UNetModel_with_MFF_MCA, UNetModel_without_MFF_MCA

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        image_size=64,
        biased_initialization = False,  # 偏置初始化-范式P2
        weight_path_similarity = 0.0,
        use_MFF_MAC = False,
        num_channels=64,                # 模型每一层的基础通道数，乘以 channel_mult 后得到每一层的实际通道数，增加 num_channels 可以提升模型的表达能力，但也会增加计算资源的需求
        num_res_blocks=2,
        num_heads=2,                    # 控制多头注意力机制中头的数量，增加头的数量可以让模型在不同的子空间中学习不同的特征表示，从而提升模型的表达能力和性能
        num_heads_upsample=-1,
        attention_resolutions="8, 4",   # 控制U-Net中的 ResBlock+AttenBlock 在哪些分辨率（如 16x16, 8x8）下开启注意力
        dropout=0.0,                    # dropout 率，增加 dropout 可以帮助模型防止过拟合，但过高的 dropout 率可能会导致模型欠拟合
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,           # 扩散过程的总步数
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,           # 控制模型的输出目标，如果 predict_xstart 为 False，则模型预测噪声 epsilon；如果 predict_xstart 为 True，则模型直接预测原始图像 x_0，这两种方式在训练和采样过程中会有不同的表现和效果
        rescale_timesteps=True,         # 是否对时间步长进行缩放，通常在训练过程中会将时间步长缩放到 [0, 1000] 的范围内，以帮助模型更好地学习不同时间步长下的特征表示
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    biased_initialization,
    weight_path_similarity,
    use_MFF_MAC,
):
    model = create_model(
        image_size,
        num_channels,
        use_MFF_MAC,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        biased_initialization = biased_initialization,
        weight_path_similarity= weight_path_similarity,
        use_MFF_MAC = use_MFF_MAC,
    )
    return model, diffusion


def create_model(
    image_size,
    num_channels,
    use_MFF_MAC,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)   # 中间层分辨率为 8x8
    elif image_size == 128:
        channel_mult = (1, 2, 3, 4, 4)      # 中间层分辨率为 8x8
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)         # 中间层分辨率为 8x8
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)         # 中间层分辨率为 4x4
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # 提取公共参数字典，避免重复写两遍
    model_kwargs = dict(
        in_channels=1 if use_MFF_MAC else 4,                    # 输入图像的通道数，输入为 M_o 或 M_o+M_r
        model_channels=num_channels,                            # mc：模型每一层的基础通道数，乘以channel_mult后得到每一层的实际通道数，中间层的输入输出通道数不变：num_channels * channel_mult[-1]
        out_channels=(1 if not learn_sigma else 2),             # 输出通道数，如果 learn_sigma 为 False，则输出3通道的图像；如果 learn_sigma 为 True，则输出6通道，其中前3通道用于预测图像，后3通道用于预测噪声的方差
        num_res_blocks=num_res_blocks,                          # nr：每个分辨率下的残差块数量  
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,                              # 控制了不同层编码器和解码器的channel数：当前层通道数 = model_channels * mult
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

    if use_MFF_MAC:
        return UNetModel_with_MFF_MCA(**model_kwargs)
    else:
        return UNetModel_without_MFF_MCA(**model_kwargs)


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    biased_initialization = False,
    weight_path_similarity = 0.0,
    use_MFF_MAC = False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    # 根据开关选择不同的类
    if use_MFF_MAC:
        diffusion_class = SpacedDiffusion
    else:
        diffusion_class = SpacedDiffusion_without_MFF_MCA

    return diffusion_class(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        biased_initialization=biased_initialization,
        weight_path_similarity = weight_path_similarity,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

import torch as th

def sample_filter(sample, Path_inverse=False, threshold_255=50):
    """
    专门用于将扩散模型输出映射到 [0, 255] 过滤噪声，并还原回 [-1, 1]
    
    Args:
        sample (torch.Tensor): 扩散模型的原始输出, 理论范围 [-1, 1]
        Path_inverse (bool): 
            True:  0 是路径, 255 是背景 (噪声是靠近 0 的浅色点)
            False: 255 是路径, 0 是背景 (噪声是靠近 0 的深色点)
        threshold_255 (int): 过滤阈值 [0, 255]
        
    Returns:
        torch.Tensor: 过滤后的张量, 范围 [-1, 1]
    """
    # 映射到 [0, 255]
    sample_255 = ((sample + 1) * 127.5).clamp(0, 255)

    if Path_inverse:
        # 路径是 0。判定范围: [0, threshold_255]
        # 只要足够“黑”，就是路径
        sample_bin = th.where(sample_255 < threshold_255, 0.0, 255.0)
    else:
        # 路径是 255。判定范围: [255 - threshold_255, 255]
        # 只要足够“白”，就是路径
        sample_bin = th.where(sample_255 > (255 - threshold_255), 255.0, 0.0)
        
    # 还原到 [-1, 1]
    # 公式: x_norm = (x_255 / 127.5) - 1
    sample_filtered = (sample_bin / 127.5) - 1.0
    
    return sample_filtered