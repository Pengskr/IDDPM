import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        sample_diffusion,
        sample_Mo,
        sample_Mr,
        sample_P,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.sample_diffusion = sample_diffusion
        self.sample_Mo = sample_Mo.to(dist_util.dev())
        self.sample_Mr = sample_Mr.to(dist_util.dev())
        self.sample_P = sample_P.to(dist_util.dev())
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0    # 断点续传步数
        self.global_batch = self.batch_size * dist.get_world_size()

        # 新增：用于绘图的历史记录
        self.mse_history = []
        self.test_loss_history = []      # 新增：model 测试损失
        self.ema_test_loss_history = []  # 新增：ema_model 测试损失
        self.step_history = []

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()    # 获取续训信息：网络参数 以及 resume_step
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)    # AdamW 修正了 Adam 在权重衰减（Weight Decay）实现上的一个数学错误。
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps                                    # 无限模式：如果 lr_anneal_steps 设为 0（默认通常如此），循环将永远运行下去，直到你手动停止。
            or self.step + self.resume_step < self.lr_anneal_steps      # 有限模式：如果你设定了具体lr_anneal_steps（例如 500,000 步），它会计算 当前步数 + 已训步数 是否达到了目标。
        ):
            M_o, M_r, P_i, cond, _ = next(self.data)                         # 从数据生成器中获取一个批次的训练数据和对应的条件信息（如果有的话）。
            self.run_step(M_o, M_r, P_i, cond)                               # 执行一个训练步骤，包括前向传播、反向传播和优化器更新
            if self.step % self.log_interval == 0:
                # 仅在主进程计算测试损失，避免多卡重复计算耗时
                if dist.get_rank() == 0:
                    # 计算当前模型和 EMA 模型的测试损失
                    t_loss = self.calculate_test_loss(use_ema=False)
                    ema_t_loss = self.calculate_test_loss(use_ema=True)
                    logger.logkv("test_loss", t_loss)
                    logger.logkv("ema_test_loss", ema_t_loss)
                
                # 统一转储 (此时缓冲区包含训练 mse, step, samples, 以及刚刚存入的 test_loss)
                report = logger.dumpkvs()

                # 更新绘图历史记录 (从 report 中提取已平均或记录的值)
                if dist.get_rank() == 0:
                    self.step_history.append(self.step + self.resume_step)
                    
                    if "mse" in report:
                        self.mse_history.append(report["mse"])
                    if "test_loss" in report:
                        self.test_loss_history.append(report["test_loss"])
                    if "ema_test_loss" in report:
                        self.ema_test_loss_history.append(report["ema_test_loss"])
                    
                    # 绘制曲线
                    self.plot_metrics(self.step + self.resume_step)

            if self.step % self.save_interval == 0:
                self.save()                                             # 每隔 save_interval 步，保存模型检查点。
                # 在保存模型时进行一次采样观察
                if dist.get_rank() == 0:
                    self.log_samples(self.sample_Mo, self.sample_Mr, self.sample_P) # 使用当前 batch 的地图作为条件

                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def calculate_test_loss(self, use_ema=False):
        """计算生成的路径 P 与真实路径 sample_P 的相似度损失 (MSE)"""
        self.model.eval()
        
        # 如果使用 EMA，暂时替换参数
        if use_ema:
            online_backup = [p.data.clone() for p in self.model.parameters()]
            self._load_ema_to_model()   
            
        with th.no_grad():
            # 使用 DDIM 采样生成路径
            model_kwargs = {}
            gen_P = self.sample_diffusion.ddim_sample_loop(
                self.model,
                self.sample_Mo,
                self.sample_Mr,
                self.sample_P.shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
            )
            # 计算路径相似度损失 (MSE)
            loss_tensor = self.sample_diffusion.loss_path_similarity(self.sample_P, gen_P)
            loss = loss_tensor.mean().detach().cpu().item()

        # 恢复原始参数
        if use_ema:
            for p, backup_val in zip(self.model.parameters(), online_backup):
                p.data.copy_(backup_val)
                
        self.model.train()
        return loss

    def _load_ema_to_model(self):
        """辅助方法：将 EMA 参数加载到当前模型中"""
        for p, ema_p in zip(self.model.parameters(), self.ema_params[0]):
            p.data.copy_(ema_p.data)    # 直接在原地修改张量的数值，而不破坏计算图

    def plot_metrics(self, step):
        """更新后的绘图函数，展示训练和测试对比"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.step_history, self.mse_history, label='Train Loss (MSE: Epsilon+Path)', alpha=0.6)
        plt.plot(self.step_history, self.test_loss_history, label='Test Loss (Model)', linewidth=2)
        plt.plot(self.step_history, self.ema_test_loss_history, label='Test Loss (EMA)', linewidth=2, linestyle='--')
        
        plt.xlabel('Step')
        plt.ylabel('Loss/MSE')
        plt.title(f'Training Progress (Step {step})')
        plt.legend()
        plt.yscale('log') # 路径损失通常较小，建议用对数坐标
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plot_path = os.path.join(get_blob_logdir(), f"metrics_curve_{step}.png")
        plt.savefig(plot_path)
        plt.close()         

    # 在 TrainLoop 类中新增方法
    def log_samples(self, M_o, M_r, P):
        self.model.eval() # 切换到评估模式
        with th.no_grad():                       
            batch_size = M_o.shape[0]
            sets_per_row = 4  # 每行显示 4 组
            num_cols_per_set = 3 # 每组包含 3 张图 (M_o, P, Sample)
            num_rows = (batch_size + sets_per_row - 1) // sets_per_row
            num_cols = sets_per_row * num_cols_per_set # 总列数为 12
            
            # 使用 DDIM 采样生成路径
            model_kwargs = {}
            sample = self.sample_diffusion.ddim_sample_loop(
                self.model,
                M_o.to(dist_util.dev()), 
                M_r.to(dist_util.dev()), 
                M_o.shape,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
            )
            
            # 可视化前处理：恢复到 0-255 像素范围并转为 Numpy
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            M_r_cpu = ((M_r + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            # 同样处理真实路径 P
            P_cpu = ((P + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
            
            # 创建画布：宽设为 12 列，高根据行数动态调整
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 2.5))
            
            # 确保 axes 始终是 2D 数组方便索引 [row, col]
            if num_rows == 1:
                axes = axes[None, :]
            
            for i in range(batch_size):
                row = i // sets_per_row
                set_idx = i % sets_per_row
                col_start = set_idx * num_cols_per_set
                
                # 1. 绘制地图 M_r
                ax_mr = axes[row, col_start]
                ax_mr.imshow(M_r_cpu[i].transpose(1, 2, 0)) # 注意：这里不需要 cmap='gray'
                if row == 0: ax_mr.set_title("M_r to model", fontsize=8)
                ax_mr.axis('off')                
                
                # 2. 绘制真实路径 P
                ax_p = axes[row, col_start + 1]
                ax_p.imshow(P_cpu[i, 0], cmap='gray')
                ax_p.axis('off')
                if row == 0: ax_p.set_title("P (GT)", fontsize=8)
                
                # 3. 绘制生成路径 Sample
                ax_s = axes[row, col_start + 2]
                ax_s.imshow(sample[i, 0], cmap='gray')
                ax_s.axis('off')
                if row == 0: ax_s.set_title("Sample", fontsize=8)
            
            # 隐藏多余的空白子图
            for i in range(batch_size * num_cols_per_set, num_rows * num_cols):
                r, c = divmod(i, num_cols)
                axes[r, c].axis('off')
            
            plt.tight_layout(pad=0.5)
            
            out_path = os.path.join(get_blob_logdir(), f"batch_sample_{self.step + self.resume_step:06d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        self.model.train() # 恢复到训练模式

    def run_step(self, M_o, M_r, P_i, cond):
        self.forward_backward(M_o, M_r, P_i, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, M_o, M_r, P_i, cond):
        zero_grad(self.model_params)
        for i in range(0, M_o.shape[0], self.microbatch):
            micro_M_o = M_o[i : i + self.microbatch].to(dist_util.dev())
            micro_M_r = M_r[i : i + self.microbatch].to(dist_util.dev())
            micro_P_i = P_i[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= M_o.shape[0]
            t, weights = self.schedule_sampler.sample(micro_M_o.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses, # 进行一次前向过程和神经网络预测，计算损失函数，返回一个字典，包含了不同类型的损失（如 "loss", "mse", "vb" 等），这些损失可以用于监控训练过程中的性能和收敛情况
                self.ddp_model,
                micro_M_o,
                micro_M_r,
                micro_P_i,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        '''
        计算模型所有参数梯度的 L2 范数，用于监控训练稳定性（如果该值过大，可能发生梯度爆炸）
        '''
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}")
                if not rate:
                    # filename = f"model{(self.step+self.resume_step):06d}.pt"
                    filename = f"model.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        #     with bf.BlobFile(
        #         bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
        #         "wb",
        #     ) as f:
        #         th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    '''
    将整个扩散过程的 $T$ 个时间步平均分为 4 个区间（Quartiles），分别统计每个区间内的平均损失。
    q0: $t \in [0, 0.25T)$（接近原始图像，去噪最容易）
    q1: $t \in [0.25T, 0.5T)$
    q2: $t \in [0.5T, 0.75T)$
    q3: $t \in [0.75T, T]$（接近纯噪声，去噪最难）

    q0 (低噪声区间，$t$ 接近 0)：这个阶段图像几乎是清晰的。虽然直觉上觉得容易，但由于模型预测的是噪声 $\epsilon$，在 $t$ 非常小时，信噪比（SNR）极高，微小的预测偏差在数值上可能会体现为较大的 MSE 损失。
    q3 (高噪声区间，$t$ 接近 $T$)：这个阶段图像几乎是纯噪声。模型此时的任务是从纯噪声中预测结构，虽然任务很难，但在训练后期，模型对于预测“噪声中的噪声”往往能达到一个相对稳定的平均损失水平。

    如果某个区间 Loss 远高于其他区间：说明模型在那个特定的噪声强度下还没学好。例如，如果 loss_q0 持续极高，可能意味着模型无法恢复清晰的细节。
    如果四个区间 Loss 非常接近：通常意味着您的噪声调度器设置得比较科学（如使用了余弦调度），模型在各阶段都在均衡地学习。
    '''
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
