"""
Train a diffusion model on images.
"""
import os
import datetime
import argparse

# 设置 保存模型和日志地址 的环境变量
if "OPENAI_LOGDIR" not in os.environ:
    subdir = datetime.datetime.now().strftime("run-%Y-%m-%d-%H-%M-%S")
    os.environ["OPENAI_LOGDIR"] = os.path.join(os.path.expanduser("~/DiffRP_IDDPM/my_model_checkpoints"), subdir)

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()  # 读取用户在终端（Terminal）输入的命令行参数

    dist_util.setup_dist()  # 设置分布式训练环境，确保在多GPU或多节点环境下正确地初始化通信和资源分配
    logger.configure()      # The logs and saved models will be written to a logging directory determined by the OPENAI_LOGDIR environment variable. If it is not set, then a temporary directory will be created in /tmp.

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )                       # 创建一个 U-Net 模型，并同时初始化一个 GaussianDiffusion 对象（负责加噪和采样逻辑）
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)  # 重要性采样器

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,                                  # 训练数据的生成器，提供训练过程中需要的图像和对应的条件信息（如果有的话）
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,                                 # Adam学习率
        ema_rate=args.ema_rate,                     # 指数移动平均（EMA）的衰减率，EMA是一种在训练过程中对模型参数进行平滑处理的方法，可以帮助模型在训练过程中更稳定地收敛，并且在评估和生成阶段通常能得到更好的性能
        log_interval=args.log_interval,             # 日志记录的频率，单位是训练步骤数，例如 log_interval=10 表示每训练10步记录一次日志
        save_interval=args.save_interval,           # 保存模型的频率，单位是训练步骤数，例如 save_interval=10000 表示每训练10000步保存一次模型
        resume_checkpoint=args.resume_checkpoint,   # 续训的检查点路径，如果提供了这个参数，训练将从指定的检查点继续，而不是从头开始。这对于长时间训练或在中途需要暂停训练的情况非常有用。
        use_fp16=args.use_fp16,                     # 是否使用半精度（FP16）训练，使用FP16可以减少显存占用并加速训练，但可能会导致数值不稳定，特别是在某些模型架构或训练设置下。
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,          # 重要性采样器的类型，默认是 "uniform"，表示在训练过程中均匀地采样时间步长。其他选项可能包括 "loss-second-moment"，它根据模型在不同时间步长上的损失来调整采样概率，以提高训练效率。
        weight_decay=args.weight_decay,             # AdamW优化器中的权重衰减（Weight Decay）参数，控制模型参数更新时的L2正则化强度，增加权重衰减可以帮助模型防止过拟合，但过高的权重衰减可能会导致模型欠拟合。
        lr_anneal_steps=args.lr_anneal_steps,       # 如果 lr_anneal_steps 设为 0（默认通常如此），训练循环将永远运行下去，直到你手动停止。
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="../datasets/cifar_train",
        schedule_sampler="loss-second-moment",     
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=100,
        batch_size=16,
        microbatch=-1,                  # -1 disables microbatches
        ema_rate="0.9999",              # comma-separated list of EMA values
        log_interval=10,
        save_interval=20,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
