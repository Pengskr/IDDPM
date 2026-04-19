import copy
import functools
import os
import time

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
from .losses import loss_path_similarity, compute_F1_score
from .script_util import sample_filter

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
        Path_inverse,
        weight_path_similarity,
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
        self.start_time = time.time()
        self.last_log_time = time.time()  # 用于计算局部每秒步数 (Steps Per Second)

        self.model = model
        self.diffusion = diffusion
        self.sample_diffusion = sample_diffusion
        self.Path_inverse = Path_inverse
        self.weight_path_similarity = weight_path_similarity
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
        self.F1_history = []
        self.F1_EMA_history = []
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
            M_o, M_r, P_i, cond, _ = next(self.data)                    # 从数据生成器中获取一个批次的训练数据和对应的条件信息（如果有的话）
            self.run_step(M_o, M_r, P_i, cond)                          # 执行一个训练步骤，包括前向传播、反向传播和优化器更新

            if self.step % self.log_interval == 0:
                if dist.get_rank() == 0:
                    gen_P, t_loss, f1 = self.evaluate_and_sample(use_ema=False)
                    ema_gen_P, ema_t_loss, ema_f1 = self.evaluate_and_sample(use_ema=True)
                    self.save_comparison_image(self.sample_Mr, self.sample_P, gen_P, ema_gen_P)
                    
                    self.test_loss_history.append(t_loss)
                    self.ema_test_loss_history.append(ema_t_loss)
                    self.F1_history.append(f1)
                    self.F1_EMA_history.append(ema_f1)
                
                # 统一转储 (此时缓冲区包含训练 mse, step, samples, 以及刚刚存入的 test_loss)
                report = logger.dumpkvs()
                if dist.get_rank() == 0:
                    self.step_history.append(self.step + self.resume_step)
                    if "mse" in report:
                        self.mse_history.append(report["mse"])
                    self.plot_metrics(self.step + self.resume_step)

            if self.step % self.save_interval == 0:
                self.save()                                             # 每隔 save_interval 步，保存模型检查点。
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

        if (self.step - 1) % self.save_interval != 0:
            self.save()
    
    def evaluate_and_sample(self, use_ema=False):
        self.model.eval()
        
        # 1. EMA 参数处理
        online_backup = None
        if use_ema:
            online_backup = [p.data.clone() for p in self.model.parameters()]
            self._load_ema_to_model()

        with th.no_grad():
            # 2. 执行采样 (最耗时的部分，只跑一次)
            gen_P = self.sample_diffusion.ddim_sample_loop(
                self.model,
                self.sample_Mo,
                self.sample_Mr,
                self.sample_P.shape,
                clip_denoised=True,
                device=dist_util.dev(),
            )
            
            # 3. 计算指标
            loss = loss_path_similarity(self.weight_path_similarity, self.sample_P, gen_P).mean().item()
            gen_P_filtered = sample_filter(gen_P, self.Path_inverse, threshold_255=100)

            f1 = compute_F1_score(self.sample_P, gen_P_filtered, self.Path_inverse, thresh_hold=0)
            
            # 记录到日志系统
            suffix = "(EMA)" if use_ema else ""
            logger.logkv(f"test_loss{suffix}", loss)
            logger.logkv(f"F1{suffix}", f1)

        # 4. 恢复原始参数
        if use_ema:
            for p, backup_val in zip(self.model.parameters(), online_backup):
                p.data.copy_(backup_val)
        
        self.model.train()
        
        # 返回生成结果
        return gen_P, loss, f1

    def save_comparison_image(self, M_r, P, sample, ema_sample):

        batch_size = M_r.shape[0]
        sets_per_row = 4      # 每行显示 4 组
        num_cols_per_set = 4  # 每组包含 4 张图 (Mr, P, Sample, EMA_Sample)
        num_rows = (batch_size + sets_per_row - 1) // sets_per_row
        num_cols = sets_per_row * num_cols_per_set # 总列数为 16

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
        ema_sample = ((ema_sample + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
        M_r_cpu = ((M_r + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()
        P_cpu = ((P + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().numpy()

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 2.5))
        
        if num_rows == 1:
            axes = axes[None, :]
        
        for i in range(batch_size):
            row = i // sets_per_row
            set_idx = i % sets_per_row
            col_start = set_idx * num_cols_per_set
            
            # 第一列：地图 M_r
            ax_mr = axes[row, col_start]
            ax_mr.imshow(M_r_cpu[i].transpose(1, 2, 0))
            ax_mr.axis('off')
            if row == 0: ax_mr.set_title("M_r", fontsize=8)
            
            # 第二列：真实路径 P
            ax_p = axes[row, col_start + 1]
            ax_p.imshow(P_cpu[i, 0], cmap='gray')
            ax_p.axis('off')
            if row == 0: ax_p.set_title("P (GT)", fontsize=8)
            
            # 第三列：Model 生成路径
            ax_s = axes[row, col_start + 2]
            ax_s.imshow(sample[i, 0], cmap='gray')
            ax_s.axis('off')
            if row == 0: ax_s.set_title("Model", fontsize=8)

            # 第四列：EMA Model 生成路径
            ax_ema = axes[row, col_start + 3]
            ax_ema.imshow(ema_sample[i, 0], cmap='gray')
            ax_ema.axis('off')
            if row == 0: ax_ema.set_title("EMA Model", fontsize=8)
        
        # 隐藏空白子图
        for i in range(batch_size * num_cols_per_set, num_rows * num_cols):
            r, c = divmod(i, num_cols)
            axes[r, c].axis('off')
        
        plt.tight_layout(pad=0.5)
        out_path = os.path.join(get_blob_logdir(), f"batch_sample_{self.step + self.resume_step:06d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _load_ema_to_model(self):
        """辅助方法：将 EMA 参数加载到当前模型中"""
        for p, ema_p in zip(self.model.parameters(), self.ema_params[0]):
            p.data.copy_(ema_p.data)    # 直接在原地修改张量的数值，而不破坏计算图

    def plot_metrics(self, step):
        """更新后的绘图函数，左侧显示 Loss，右侧显示 F1-score"""
        # 创建 1 行 2 列的子图布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # --- 左图：绘制 Loss 相关指标 ---
        ax1.plot(self.step_history, self.mse_history, label='Train Loss (MSE)', alpha=0.4)
        ax1.plot(self.step_history, self.test_loss_history, label='Test Loss (Model)', linewidth=2)
        ax1.plot(self.step_history, self.ema_test_loss_history, label='Test Loss (EMA)', linewidth=2, linestyle='--')
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss/MSE')
        ax1.set_title('Loss Metrics')
        ax1.legend()
        ax1.set_yscale('log')  # Loss 通常建议使用对数坐标
        ax1.grid(True, which="both", ls="-", alpha=0.3)

        # --- 右图：绘制 F1-score ---
        ax2.plot(self.step_history, self.F1_history, label='F1 (Model)', color='green', linewidth=2)
        ax2.plot(self.step_history, self.F1_EMA_history, label='F1 (EMA)', color='darkgreen', linewidth=2, linestyle='--')
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel('F1')
        ax2.set_title('F1')
        ax2.legend()
        # F1 值范围在 0-1，通常使用线性坐标，并锁定 y 轴范围
        ax2.set_ylim(0, 1.05) 
        ax2.grid(True, ls="-", alpha=0.3)

        # 整体标题
        plt.suptitle(f'Training Progress (Step {step})', fontsize=16)
        
        # 自动调整布局，防止子图重叠
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存并关闭
        plot_path = os.path.join(get_blob_logdir(), f"metrics_curve_{step}.png")
        plt.savefig(plot_path)
        plt.close()         


    def run_step(self, M_o, M_r, P_i, cond):
        self.forward_backward(M_o, M_r, P_i, cond)
        self.step += 1
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        if self.step % self.log_interval == 0: self.log_step()

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

        # self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        # 3. 核心修复：必须缩放所有主参数的梯度
        scaling_factor = 1.0 / (2 ** self.lg_loss_scale)
        for p in self.master_params:
            if p.grad is not None:
                p.grad.mul_(scaling_factor)

        if self.step % self.log_interval == 0: self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        if self.step % self.log_interval == 0: self._log_grad_norm()
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
        logger.logkv("samples", (self.step + self.resume_step) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

        # 总耗时 (Total Elapsed Time)
        elapsed_total = time.time() - self.start_time
        
        # 计算当前 log 间隔内的吞吐量 (Throughput)
        current_time = time.time()
        dt = current_time - self.last_log_time
        sps = self.log_interval / dt if dt > 0 else 0   # steps_per_sec：每秒步数，简称 SPS）是衡量深度学习训练吞吐量（Throughput）和工程效率的核心指标
        self.last_log_time = current_time

        # 写入 logger
        logger.logkv("time_elapsed_seconds", elapsed_total)
        logger.logkv("steps_per_sec", sps)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}")
                if not rate:
                    # filename = f"model{(self.step+self.resume_step):06d}.pt"
                    filename = f"model.pt"
                else:
                    # filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                    filename = f"model_ema.pt"
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
