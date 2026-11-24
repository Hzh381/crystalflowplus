"""GradNorm：晶体流匹配多任务损失的动态权重平衡。

参考
----
Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing in
Deep Multitask Networks", ICML 2018.  https://arxiv.org/abs/1711.02257

实现说明：
- 每个任务维护一个标量权重；
- 每隔 ``update_freq`` 个训练步，在共享参数上测量各任务加权损失的梯度范数；
- 收敛较慢（相对于初始损失）的任务获得更大权重，使晶格流、坐标流和
  物性预测头的训练速度保持相对均衡；
- ``update`` 会为每个活跃任务额外做一次局部反向传播，因此支持 warmup
  和 ``update_freq`` 节流。
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn


class GradNorm(nn.Module):
    def __init__(
        self,
        task_names: Sequence[str],
        alpha: float = 0.12,
        lr: float = 0.025,
        init_weights: Optional[Mapping[str, float]] = None,
        warmup_steps: int = 0,
        update_freq: int = 1,
        min_weight: float = 1e-3,
        max_weight: float = 1e3,
        eps: float = 1e-8,
    ):
        super().__init__()
        task_names = list(task_names)
        if len(task_names) < 2:
            raise ValueError(f"GradNorm needs at least two tasks, got {task_names}.")
        self.task_names = task_names
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.warmup_steps = int(warmup_steps)
        self.update_freq = max(1, int(update_freq))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)

        weights = torch.ones(len(task_names))
        if init_weights is not None:
            for i, name in enumerate(task_names):
                weights[i] = float(init_weights.get(name, 1.0))
        self.register_buffer("weights", weights)
        self.register_buffer("initial_losses", torch.zeros(len(task_names)))
        self.register_buffer("task_initialized", torch.zeros(len(task_names), dtype=torch.bool))
        self.register_buffer("last_grad_norms", torch.zeros(len(task_names)))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def combine(self, task_losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """按当前权重对各任务损失做加权求和。"""
        total = None
        for i, name in enumerate(self.task_names):
            loss = task_losses.get(name)
            if not torch.is_tensor(loss):
                continue
            # 用 float() 取出标量，避免权重 buffer 进入自动求导图：
            # update() 可能在 Lightning 调用 backward() 之前重新赋值该 buffer。
            term = loss * float(self.weights[i])
            total = term if total is None else total + term
        if total is None:
            raise ValueError(
                f"GradNorm: none of the tasks {self.task_names} produced a tensor loss."
            )
        return total

    def update(
        self,
        task_losses: Mapping[str, torch.Tensor],
        shared_params: Iterable[torch.nn.Parameter],
        step: int,
    ) -> Optional[Dict[str, float]]:
        """根据共享参数上的梯度范数更新各任务权重。

        必须在 ``training_step`` 中、前向计算图仍然存活时调用
        （即 Lightning 执行 ``loss.backward()`` 之前）。
        """
        if step < self.warmup_steps or step % self.update_freq != 0:
            return None

        params = [p for p in shared_params if p.requires_grad]
        if not params:
            return None

        idx, losses, norms = [], [], []
        for i, name in enumerate(self.task_names):
            loss = task_losses.get(name)
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            try:
                grads = torch.autograd.grad(
                    self.weights[i] * loss,
                    params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
            except RuntimeError:
                # 该任务不依赖共享参数，无法参与权重平衡，保持原权重不变。
                continue
            norm = torch.zeros((), device=self.weights.device)
            for g in grads:
                if g is not None:
                    norm = norm + g.detach().pow(2).sum()
            idx.append(i)
            losses.append(loss.detach())
            norms.append(norm.sqrt())

        if not idx:
            return None

        idx_t = torch.tensor(idx, device=self.weights.device)
        losses_t = torch.stack(losses).to(self.weights.device)
        norms_t = torch.stack(norms)

        with torch.no_grad():
            # 记录每个任务首次参与平衡时的损失，作为“逆训练速率”的参考。
            unseen = ~self.task_initialized[idx_t]
            if unseen.any():
                self.initial_losses[idx_t[unseen]] = losses_t[unseen].clamp_min(self.eps)
                self.task_initialized[idx_t[unseen]] = True

            inv_rate = losses_t / self.initial_losses[idx_t].clamp_min(self.eps)
            inv_rate = inv_rate / inv_rate.mean().clamp_min(self.eps)
            target = inv_rate * norms_t.mean()

            new_weights = self.weights.detach().clone()
            new_weights[idx_t] = (
                new_weights[idx_t] + self.lr * (target - new_weights[idx_t])
            ).clamp(self.min_weight, self.max_weight)
            total = new_weights.sum()
            if total > 0:
                new_weights = new_weights * (len(self.task_names) / total)
            # 重新绑定 buffer 而不是原地修改，避免破坏 combine() 已保存到
            # 自动求导图中的标量权重。
            self.weights = new_weights
            self.last_grad_norms = self.last_grad_norms.detach().clone().scatter(0, idx_t, norms_t)
            self.updates = self.updates + 1
            weights = {name: float(self.weights[i]) for i, name in enumerate(self.task_names)}

        return weights

    def extra_repr(self) -> str:
        tasks = ", ".join(f"{n}={float(w):.3f}" for n, w in zip(self.task_names, self.weights))
        return f"tasks=[{tasks}], alpha={self.alpha}, lr={self.lr}"
