# Training example:
#
# HYDRA_FULL_ERROR=1 python diffcsp/run.py \
# data=mp_20 model=flow \
# logging.wandb.group=mp_20 expname=flow-test-01

import copy
import math
import random
import re
import logging
from typing import Any, Dict
from functools import partial, partialmethod, wraps
from inspect import getfullargspec
from itertools import pairwise
from operator import itemgetter

import numpy as np
import omegaconf
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from tqdm import tqdm

import hydra
from diffcsp.common.data_utils import (
    EPSILON,
    cart_to_frac_coords,
    frac_to_cart_coords,
    lattice_params_to_matrix_torch,
    lattice_polar_build_torch,
    lattice_polar_decompose_torch,
    lengths_angles_to_volume,
    mard,
    min_distance_sqr_pbc,
)
from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
from diffcsp.pl_modules.hungarian import HungarianMatcher
from diffcsp.pl_modules.lattice_utils import LatticeDecompNN
# from diffcsp.pl_modules.ode_solvers import str_to_solver
from diffcsp.pl_modules.symmetrize import SymmetrizeRotavg
from diffcsp.pl_modules.conditioning import MultiEmbedding
from diffcsp.pl_modules.gradnorm import GradNorm
from diffcsp.pl_modules.time_scheduler import TimeScheduler
from diffcsp.pl_modules.type_module import TypeTableModule

MAX_ATOMIC_NUM = 100

metriclogger = logging.getLogger("metrics")


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        if hasattr(self.hparams, "model"):
            self._hparams = self.hparams.model

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial")
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.optim.lr_scheduler, optimizer=opt)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "frequency": 5, "monitor": "val_loss"}}


### Model definition


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DirectUnsqueezeTime(nn.Module):
    def forward(self, time: torch.Tensor):
        return time.unsqueeze(-1)


class CSPFlow(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.hparams.time_dim == 0:
            self.time_dim = 1
            self.time_embedding = DirectUnsqueezeTime()
        else:
            self.time_dim = self.hparams.time_dim
            self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.time_scheduler = TimeScheduler(self.hparams.get("time_scheduler", ""))

        # Classifier-Free Guidance（CFG）：cond_drop_prob 是训练时丢弃条件、
        # 学习无条件流场的概率；guide_threshold 作为旧配置名的兼容别名。
        self.guide_threshold = self.hparams.get(
            "cond_drop_prob", self.hparams.get("guide_threshold", None)
        )
        if self.guide_threshold is not None:
            cond_cfg = self.hparams.get("conditions", None)
            cond_keys = list(cond_cfg.get("cond_keys", [])) if cond_cfg is not None else []
            if len(cond_keys) == 0:
                raise ValueError(
                    "Classifier-free guidance requires at least one condition; "
                    "please set `data.conditions` or unset `cond_drop_prob`."
                )
            self.cond_emb = MultiEmbedding(**self.hparams.conditions)
            cemb_dim = self.cond_emb.n_out
        else:
            self.cond_emb = None
            cemb_dim = 1
        self.pred_type = self.hparams.decoder.get('pred_type', False)
        self.type_encoding = self.hparams.get('type_encoding', None)
        if self.type_encoding == "table":
            self.type_encoding = TypeTableModule()

        # 辅助物性预测头（研究路线 D）。
        prop_cfg = self.hparams.get("property_head", None)
        self.prop_cfg = prop_cfg
        self.prop_enabled = bool(prop_cfg is not None and prop_cfg.get("enabled", False))
        self.prop_key = None
        self.prop_supervision = "clean"
        self.prop_cost = 1.0
        self.prop_t_min = 0.0
        self.prop_power = 1.0
        self.prop_clean_freq = 1
        self.prop_hidden_dim = None
        if self.prop_enabled:
            data_cfg = self.hparams.get("data", None)
            data_props = list(data_cfg.get("properties", None) or []) if data_cfg is not None else []
            data_prop = data_cfg.get("prop", None) if data_cfg is not None else None
            self.prop_key = prop_cfg.get("key", None)
            if self.prop_key is None:
                conditions = data_cfg.get("conditions", None) if data_cfg is not None else None
                if conditions:
                    self.prop_key = conditions[0]
                else:
                    self.prop_key = data_prop or "y"
            if data_cfg is not None and self.prop_key not in data_props and self.prop_key != data_prop:
                raise ValueError(
                    f"property_head.key='{self.prop_key}' is not available in the dataset. "
                    f"Add it to `data.properties` (available: {data_props + ([data_prop] if data_prop else [])})."
                )
            self.prop_supervision = prop_cfg.get("supervision", "clean")
            if self.prop_supervision not in ("clean", "time_gated"):
                raise ValueError(
                    f"Unknown property supervision '{self.prop_supervision}', expected 'clean' or 'time_gated'."
                )
            self.prop_cost = float(prop_cfg.get("cost", 0.2))
            self.prop_t_min = float(prop_cfg.get("t_min", 0.5))
            if not 0.0 <= self.prop_t_min < 1.0:
                raise ValueError("property_head.t_min must be in [0, 1).")
            self.prop_power = float(prop_cfg.get("power", 1.0))
            self.prop_clean_freq = max(1, int(prop_cfg.get("clean_freq", 1)))
            self.prop_hidden_dim = prop_cfg.get("hidden_dim", None)

        # GradNorm 多任务动态损失平衡（研究路线 C）。
        gn_cfg = self.hparams.get("gradnorm", None)
        self.gradnorm_cfg = gn_cfg
        self.gradnorm = None
        if gn_cfg is not None and gn_cfg.get("enabled", False):
            tasks = list(gn_cfg.get("tasks", ["lattice", "coord"]))
            if self.prop_enabled and "property" not in tasks:
                tasks.append("property")
            init_weights = gn_cfg.get("init_weights", None)
            self.gradnorm = GradNorm(
                task_names=tasks,
                alpha=float(gn_cfg.get("alpha", 0.12)),
                lr=float(gn_cfg.get("lr", 0.025)),
                init_weights=dict(init_weights) if init_weights is not None else None,
                warmup_steps=int(gn_cfg.get("warmup_steps", 0)),
                update_freq=int(gn_cfg.get("update_freq", 1)),
                min_weight=float(gn_cfg.get("min_weight", 1e-3)),
                max_weight=float(gn_cfg.get("max_weight", 1e3)),
            )
        self._gn_shared_params = None

        self.decoder = hydra.utils.instantiate(
            self.hparams.decoder,
            type_encoding=self.type_encoding,
            latent_dim=self.hparams.latent_dim + self.time_dim,  # 0 + time
            cemb_dim=cemb_dim,
            pred_prop=self.prop_enabled,
            prop_hidden_dim=self.prop_hidden_dim,
            _recursive_=False,
        )
        if self.prop_enabled and not getattr(self.decoder, "pred_prop", False):
            raise ValueError(
                "property_head is only supported by decoders that implement `pred_prop` (e.g. cspnet)."
            )
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        self.ot = self.hparams.get("ot", False)
        self.permute_l = HungarianMatcher("norm")
        self.permute_f = HungarianMatcher("norm_mic")
        self.lattice_polar = self.hparams.get("lattice_polar", False)
        self.lattice_polar_sigma = self.hparams.get("lattice_polar_sigma", 1.0)
        self.latticedecompnn = LatticeDecompNN()
        self.from_cubic = self.hparams.get("from_cubic", False)
        self.lattice_teacher_forcing = self.hparams.get("lattice_teacher_forcing", -1)
        self.symmetrize_anchor = self.hparams.get("symmetrize_anchor", False)
        self.symmetrize_rotavg = self.hparams.get("symmetrize_rotavg", False)
        self.post_symmetrize = self.hparams.get("post_symmetrize", True)
        self.symm_rotavg = SymmetrizeRotavg()
        self.use_symmetrize_loss = self.hparams.get("use_symmetrize_loss", False)
        self.cost_sym_lattice = self.hparams.get("cost_sym_lattice", self.hparams.cost_lattice)
        self.cost_sym_coord = self.hparams.get("cost_sym_coord", self.hparams.cost_coord)

        if self.ot:
            hydra.utils.log.info("Using optimal transport")
        if self.lattice_polar:
            hydra.utils.log.info(f"Using lattice polar decomposition with sigma={self.lattice_polar_sigma}")
        if self.from_cubic:
            hydra.utils.log.info("Using cubic lattice")
        if self.lattice_teacher_forcing > 0:
            hydra.utils.log.info(f"Using lattice_teacher_forcing={self.lattice_teacher_forcing}")
        if self.symmetrize_anchor:
            hydra.utils.log.info("Using symmetrize_anchor")
        if self.symmetrize_rotavg:
            hydra.utils.log.info("Using symmetrize_rotavg")
        if self.symmetrize_anchor and self.symmetrize_rotavg:
            raise ValueError("You can only specify one symmetrize method from anchor|rotavg")
        if self.keep_lattice:
            hydra.utils.log.warning(f"cost_lattice={self.hparams.cost_lattice}, setting to keep lattice.")
        if self.keep_coords:
            hydra.utils.log.warning(f"cost_coords={self.hparams.cost_coord}, setting to keep coords.")
        if self.guide_threshold is not None:
            hydra.utils.log.info(
                f"Classifier-free guidance enabled (cond_drop_prob={self.guide_threshold})."
            )
        if self.prop_enabled:
            hydra.utils.log.info(
                f"Property head enabled: key={self.prop_key}, cost={self.prop_cost}, "
                f"supervision={self.prop_supervision}."
            )
        if self.gradnorm is not None:
            hydra.utils.log.info(f"GradNorm enabled: {self.gradnorm}")

    def sample_lengths(self, num_atoms, batch_size):
        loc = math.log(2)
        scale = math.log(1)
        lengths = torch.randn((batch_size, 3), device=self.device)
        lengths = torch.exp(lengths * scale + loc)
        lengths = num_atoms[:, None] * lengths
        return lengths

    def sample_angles(self, num_atoms, batch_size):
        angles = torch.rand((batch_size, 3), device=self.device)  # [60, 120)
        angles = angles * 60 + 60
        return angles

    def sample_lattice(self, batch_size):
        l0 = torch.randn([batch_size, 3, 3], device=self.device)
        return l0

    def sample_lattice_polar(self, batch_size):
        l0 = torch.randn([batch_size, 6], device=self.device) * self.lattice_polar_sigma
        l0[:, -1] = l0[:, -1] + 1
        if self.from_cubic:
            l0[:, :5] = 0
        return l0

    def _property_target(self, batch, ref):
        """取出 batch 中已归一化的目标物性；缺失时回退到 batch.y。"""
        if self.prop_key == "y":
            target = batch.y
        else:
            target = batch.get(self.prop_key)
            if target is None:
                target = batch.y
        return target.view_as(ref)

    def _property_loss(self, pred_prop, batch, times):
        target = self._property_target(batch, pred_prop)
        per_graph = F.mse_loss(pred_prop, target, reduction="none").mean(dim=-1)
        if self.prop_supervision == "time_gated":
            # 只对接近真实结构的时间步（t >= t_min）做监督，并按 t^power 加权，
            # 避免把任意噪声中间态当作真实物性目标。
            if self.prop_t_min > 0.0:
                gate = ((times - self.prop_t_min) / (1.0 - self.prop_t_min)).clamp(0.0, 1.0)
            else:
                gate = torch.ones_like(times)
            gate = gate.pow(self.prop_power)
            return (per_graph * gate).sum() / gate.sum().clamp_min(1.0)
        return per_graph.mean()

    def _clean_property_prediction(self, batch, frac_coords, lattices_rep, lattices_mat):
        """在真实（t=1）结构上做物性预测。

        预测头不应看到噪声中间态，也不应看到条件向量，否则会学成“复述
        目标物性”而不是根据结构进行估计。
        """
        times = self.time_scheduler(torch.ones(batch.num_graphs, device=self.device))
        time_emb = self.time_embedding(times)
        if self.pred_type:
            if getattr(self.decoder, "smooth", True):
                if self.type_encoding is None:
                    atom_types = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
                else:
                    atom_types = self.type_encoding(batch.atom_types)
            else:
                atom_types = batch.atom_types
        else:
            atom_types = batch.atom_types
        outs = self.decoder(
            t=time_emb,
            atom_types=atom_types,
            frac_coords=frac_coords,
            lattices_rep=lattices_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=lattices_mat,
            cemb=None,
            guide_indicator=None,
        )
        return outs[-1]

    @torch.no_grad()
    def predict_property(self, batch):
        """对 batch 中的真实晶体结构预测（已归一化的）辅助物性。"""
        if not self.prop_enabled:
            raise RuntimeError("This model has no property head (property_head.enabled=False).")
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        if self.lattice_polar:
            lattices_rep = lattice_polar_decompose_torch(lattices)
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                lattices_rep = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep, batch.spacegroup)
            lattices_mat = lattice_polar_build_torch(lattices_rep)
        else:
            lattices_rep = lattices
            lattices_mat = lattices
        return self._clean_property_prediction(
            batch, batch.frac_coords, lattices_rep, lattices_mat
        )

    def forward(self, batch, guide_threshold=None):

        batch_size = batch.num_graphs
        times = torch.rand(batch_size, device=self.device)
        times = self.time_scheduler(times)
        time_emb = self.time_embedding(times)

        guide_threshold = self.guide_threshold if guide_threshold is None else guide_threshold
        if guide_threshold is None:
            cemb = None
            guide_indicator = None
        else:
            if self.cond_emb is None:
                raise Exception("Model is not initialized with guidance")
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = torch.rand(batch_size, device=self.device) - guide_threshold
            guide_indicator = guide_indicator.heaviside(torch.tensor(1.0, device=self.device))

        # Build time stamp T and 0
        # lattice
        if self.lattice_polar:
            lattices_rep_T = batch.lattice_polar
            lattices_rep_0 = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                lattices_rep_T = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_T, batch.spacegroup)
                lattices_rep_0 = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_0, batch.spacegroup)
            elif self.symmetrize_rotavg:
                lattices_rep_T = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_T, batch.spacegroup)
                lattices_rep_0 = self.latticedecompnn.proj_k_to_spacegroup(lattices_rep_0, batch.spacegroup)
            lattices_mat_T = lattice_polar_build_torch(lattices_rep_T)
        else:
            lattices_rep_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            lattices_rep_0 = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_T = lattices_rep_T
        # lattice teacher forcing
        lattice_teacher_forcing = self.current_epoch < self.lattice_teacher_forcing
        if lattice_teacher_forcing:
            lattices_rep_0 = lattices_rep_T

        # coords
        frac_coords = batch.frac_coords
        f0 = torch.rand_like(frac_coords)
        if self.symmetrize_anchor:
            f0_anchor = f0[batch.anchor_index]
            f0_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index], f0_anchor)
            f0 = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f0_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f0 = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f0,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        # types
        if self.pred_type:
            if self.type_encoding is None:
                gt_atom_types_onehot = F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM).float()
                rd_atom_types_onehot = torch.randn_like(gt_atom_types_onehot)
            else:
                gt_atom_types_onehot = self.type_encoding(batch.atom_types)
                rd_atom_types_onehot = self.type_encoding.get_rd_encoded_types(batch.num_nodes, device=self.device)

        # optimal transport
        if self.ot:
            if self.symmetrize_anchor or self.symmetrize_rotavg:
                raise ValueError("OT is forbidden in symmetrize.")
            _, lattices_rep_0 = self.permute_l(lattices_rep_T, lattices_rep_0)
            _, f0 = self.permute_f(frac_coords, f0)

        # Build time stamp t
        tar_l = lattices_rep_T - lattices_rep_0
        tar_f = (frac_coords - f0 - 0.5) % 1 - 0.5
        if self.pred_type:
            tar_t = gt_atom_types_onehot - rd_atom_types_onehot

        # Build input lattice rep/mat and input coords
        l_expand_dim = (slice(None),) + (None,) * (tar_l.dim() - 1)
        input_lattice_rep = lattices_rep_0 + times[l_expand_dim] * tar_l
        input_frac_coords = f0 + times.repeat_interleave(batch.num_atoms)[:, None] * tar_f
        if self.lattice_polar:
            input_lattice_mat = lattice_polar_build_torch(input_lattice_rep)
        else:
            input_lattice_mat = input_lattice_rep
        if self.pred_type:
            input_atom_type_probs = rd_atom_types_onehot + times.repeat_interleave(batch.num_atoms)[:, None] * tar_t
            input_atom_types = input_atom_type_probs
        else:
            input_atom_types = batch.atom_types

        # Replace inputs if fixed
        if self.keep_coords:
            input_frac_coords = frac_coords
        if self.keep_lattice:
            input_lattice_rep = lattices_rep_T
            input_lattice_mat = lattices_mat_T

        # Flow
        pred = self.decoder(
            t=time_emb,
            atom_types=input_atom_types,
            frac_coords=input_frac_coords,
            lattices_rep=input_lattice_rep,
            num_atoms=batch.num_atoms,
            node2graph=batch.batch,
            lattices_mat=input_lattice_mat,
            cemb=cemb, guide_indicator=guide_indicator,
        )
        pred_l, pred_f = pred[0], pred[1]
        out_idx = 2
        if self.pred_type:
            pred_t = pred[out_idx]
            out_idx += 1
        if self.prop_enabled:
            pred_prop = pred[out_idx]

        loss_sym_l = 0.0
        loss_sym_f = 0.0
        if self.post_symmetrize and self.symmetrize_anchor:
            if self.lattice_polar:
                pred_l_symmetrized = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            tar_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, tar_f)
            pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
            pred_f_symmetrized = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor)
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_symmetrized)
                loss_sym_f = F.mse_loss(pred_f, pred_f_symmetrized)
            # for loss
            pred_l = pred_l_symmetrized
            tar_f = tar_f_anchor
            pred_f = pred_f_anchor
        elif self.post_symmetrize and self.symmetrize_rotavg:
            if self.lattice_polar:
                pred_l_symmetrized = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f_symmetrized = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=pred_f,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            )
            if self.use_symmetrize_loss:
                loss_sym_l = F.mse_loss(pred_l, pred_l_symmetrized)
                loss_sym_f = F.mse_loss(pred_f, pred_f_symmetrized)
            pred_l = pred_l_symmetrized
            pred_f = pred_f_symmetrized

        loss_lattice = F.mse_loss(pred_l, tar_l)
        loss_coord = F.mse_loss(pred_f, tar_f)
        if self.pred_type:
            loss_type = F.mse_loss(pred_t, tar_t)
        else:
            loss_type = 0.0

        loss_prop = 0.0
        if self.prop_enabled:
            # clean：在真实结构上额外前向一次；time_gated：复用当前噪声态并按时间门控。
            if self.prop_supervision == "clean":
                run_clean = (not self.training) or (self.global_step % self.prop_clean_freq == 0)
                if run_clean:
                    pred_prop_clean = self._clean_property_prediction(
                        batch, frac_coords, lattices_rep_T, lattices_mat_T
                    )
                    loss_prop = self._property_loss(
                        pred_prop_clean, batch, torch.ones(batch_size, device=self.device)
                    )
            else:
                loss_prop = self._property_loss(pred_prop, batch, times)

        cost_coord = self.hparams.cost_coord
        cost_lattice = 0.0 if lattice_teacher_forcing else self.hparams.cost_lattice
        cost_type    = 0.0 if not self.pred_type      else self.hparams.cost_type
        loss_terms = {}
        if cost_lattice > 0.0 or self.cost_sym_lattice > 0.0:
            loss_terms['lattice'] = cost_lattice * loss_lattice + self.cost_sym_lattice * loss_sym_l
        if cost_coord > 0.0 or self.cost_sym_coord > 0.0:
            loss_terms['coord'] = cost_coord * loss_coord + self.cost_sym_coord * loss_sym_f
        if self.pred_type and cost_type > 0.0:
            loss_terms['type'] = cost_type * loss_type
        if self.prop_enabled:
            loss_terms['property'] = self.prop_cost * loss_prop
        if not loss_terms:
            raise RuntimeError("All flow-matching loss weights are zero.")
        loss = sum(loss_terms.values())

        return {
            'loss': loss,
            'loss_terms': loss_terms,
            'loss_lattice': loss_lattice,
            'loss_coord': loss_coord,
            'loss_type': loss_type,
            'loss_property': loss_prop,
            'loss_sym_lattice': loss_sym_l,
            'loss_sym_coord': loss_sym_f,
        }

    def _get_gradnorm_params(self):
        """GradNorm 测量梯度范数所用的共享参数。"""
        if self._gn_shared_params is not None:
            return self._gn_shared_params
        params = None
        path = None
        if self.gradnorm_cfg is not None:
            path = self.gradnorm_cfg.get("shared_module", None)
        if path:
            module = self
            for part in path.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is not None:
                params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            names = dict(self.named_parameters())
            matched = [n for n in names if n.startswith("decoder.csp_layer_")]
            if matched:
                def _layer_idx(name):
                    m = re.search(r"csp_layer_(\d+)", name)
                    return int(m.group(1)) if m else -1

                last = max(_layer_idx(n) for n in matched)
                params = [p for n, p in names.items() if _layer_idx(n) == last]
            else:
                params = [p for p in self.decoder.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("GradNorm could not resolve any shared parameters.")
        self._gn_shared_params = params
        return params

    @staticmethod
    def get_anneal_factor(t, slope: float = 0.0, offset: float = 0.0):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        return 1 + slope * F.relu(t - offset)

    def post_decoder_on_sample(
        self, pred,
        batch, t,
        anneal_lattice=False, anneal_coords=False, anneal_type=False,
        anneal_slope=0.0, anneal_offset=0.0,
    ):
        pred_l, pred_f = pred[0], pred[1]
        if self.pred_type:
            pred_t = pred[2]
        if self.symmetrize_anchor:
            if self.lattice_polar:
                pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f_anchor = torch.einsum('bij,bj->bi', batch.ops_inv, pred_f)
            pred_f_anchor = scatter(pred_f_anchor, batch.anchor_index, dim=0, reduce = 'mean')[batch.anchor_index]
            pred_f = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], pred_f_anchor)
        elif self.symmetrize_rotavg:
            if self.lattice_polar:
                pred_l = self.latticedecompnn.proj_kdiff_to_spacegroup(pred_l, batch.spacegroup)
            else:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            pred_f = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=pred_f,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            )
        anneal_factor = self.get_anneal_factor(t, anneal_slope, anneal_offset)
        if anneal_lattice:
            pred_l *= anneal_factor
        if anneal_coords:
            pred_f *= anneal_factor
        if self.pred_type:
            if anneal_type:
                pred_t *= anneal_factor
            return pred_l, pred_f, pred_t
        else:
            return pred_l, pred_f

    @torch.no_grad()
    def sample(
        self, batch, step_lr=None, N=None,
        anneal_lattice=False, anneal_coords=False, anneal_type=False,
        anneal_slope=0.0, anneal_offset=0.0,
        guide_factor=None,
        guide_s_min=None, guide_s_max=None,
        guide_gamma=1.0, guide_time_dir="ascending",
        **kwargs,
    ):
        """
        Classifier-Free Guidance 采样::

            v_cfg(t) = v_uncond + s(t) * (v_cond - v_uncond)

        未指定时间调度时，``guide_factor`` 即为常数引导强度。给定
        ``guide_s_min`` / ``guide_s_max`` 时，强度服从
        ``s(t) = s_min + (s_max - s_min) * u**guide_gamma``，其中
        ``guide_time_dir="ascending"`` 时 ``u = t``，``"descending"`` 时
        ``u = 1 - t``（t=0 为噪声端，t=1 为数据端）。

        Requires
            non-symmetry: num_nodes atom_types
            symmetry: num_nodes atom_types spacegroup anchor_index ops ops_inv (general_ops symm_map num_general_ops)
        """
        if N is None:
            N = round(1 / step_lr)

        batch_size = batch.num_graphs

        if guide_time_dir not in ("ascending", "descending"):
            raise ValueError(
                f"guide_time_dir must be 'ascending' or 'descending', got {guide_time_dir!r}."
            )
        use_guidance = self.guide_threshold is not None
        if not use_guidance and guide_factor is not None:
            raise ValueError("Model is not trained with guidance but trying to sample with guidance.")
        if use_guidance and guide_factor is None:
            if guide_s_min is None and guide_s_max is None:
                raise ValueError("Model is trained with guidance but trying to sample with no guidance.")
            guide_factor = 1.0
        if use_guidance:
            if self.cond_emb is None:
                raise Exception("Model is not trained with guidance")
            missing = [key for key in self.cond_emb.cond_keys if batch.get(key) is None]
            if missing:
                raise ValueError(
                    f"Guided sampling requires the conditions {missing}; "
                    "pass them through the sampling batch (e.g. --conditions)."
                )
            cemb = self.cond_emb(**{key: batch.get(key) for key in self.cond_emb.cond_keys})
            guide_indicator = torch.ones(batch_size, device=self.device)

        # time stamp T
        # lattice
        if self.lattice_polar:
            l_T = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                l_T = self.latticedecompnn.proj_k_to_spacegroup(l_T, batch.spacegroup)
            elif self.symmetrize_rotavg:
                l_T = self.latticedecompnn.proj_k_to_spacegroup(l_T, batch.spacegroup)
            lattices_mat_T = lattice_polar_build_torch(l_T)
        else:
            l_T = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_T = l_T
        # coords
        f_T = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.symmetrize_anchor:
            f_T_anchor = f_T[batch.anchor_index]
            f_T_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_T_anchor)
            f_T = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_T_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f_T = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f_T,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]
        # types
        if self.pred_type:
            if self.type_encoding is None:
                rd_atom_types_onehot = torch.randn((batch.num_nodes, MAX_ATOMIC_NUM), device=self.device)
                atom_types = torch.argmax(rd_atom_types_onehot, dim=-1) + 1
            else:
                rd_atom_types_onehot = self.type_encoding.get_rd_encoded_types(batch.num_nodes, device=self.device)
                atom_types = self.type_encoding.decode_types(rd_atom_types_onehot)
        else:
            atom_types = batch.atom_types

        #
        if self.keep_coords:
            f_T = batch.frac_coords
        if self.keep_lattice:
            lattices_mat_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
            if self.lattice_polar:
                l_T = lattice_polar_decompose_torch(lattices_mat_T)
            else:
                l_T = lattices_mat_T

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types,
                'frac_coords': f_T % 1.0,
                'lattices': lattices_mat_T,
            }
        }

        lattices_mat_t = lattices_mat_T.clone().detach()
        l_t = l_T.clone().detach()
        f_t = f_T.clone().detach()
        if self.pred_type:
            t_t = rd_atom_types_onehot.clone().detach()
        else:
            t_t = batch.atom_types


        for t in tqdm(range(1, N + 1)):

            t_stamp = t / N
            times = torch.full((batch_size,), t_stamp, device=self.device)
            time_emb = self.time_embedding(times)

            if self.keep_coords:
                f_t = f_T
            if self.keep_lattice:
                l_t = l_T
                lattices_mat_t = lattices_mat_T

            # ========= pred each step start =========
            if use_guidance:
                # 计算当前时间步的引导强度 s(t)。
                strength = guide_factor
                if guide_s_min is not None or guide_s_max is not None:
                    s_min = guide_factor if guide_s_min is None else guide_s_min
                    s_max = guide_factor if guide_s_max is None else guide_s_max
                    u = t_stamp if guide_time_dir == "ascending" else 1.0 - t_stamp
                    strength = s_min + (s_max - s_min) * (u ** guide_gamma)

                def _decode(use_cond):
                    pred = self.decoder(
                        t=time_emb,
                        atom_types=t_t,
                        frac_coords=f_t,
                        lattices_rep=l_t,
                        num_atoms=batch.num_atoms,
                        node2graph=batch.batch,
                        lattices_mat=lattices_mat_t,
                        cemb=cemb if use_cond else None,
                        guide_indicator=guide_indicator if use_cond else None,
                    )
                    return self.post_decoder_on_sample(
                        pred,
                        batch=batch, t=t_stamp,
                        anneal_lattice=anneal_lattice, anneal_coords=anneal_coords,
                        anneal_type=anneal_type,
                        anneal_slope=anneal_slope, anneal_offset=anneal_offset,
                    )

                if abs(strength - 1.0) < 1e-4:
                    # s≈1 时结果等于条件预测，跳过无条件分支以省算力。
                    pred = _decode(True)
                elif abs(strength) < 1e-4:
                    # s≈0 时结果等于无条件预测，跳过条件分支。
                    pred = _decode(False)
                else:
                    pred_cond = _decode(True)
                    pred_uncond = _decode(False)
                    pred = tuple(
                        u_v + strength * (c_v - u_v)
                        for u_v, c_v in zip(pred_uncond, pred_cond)
                    )
            else:
                pred = self.decoder(
                    t=time_emb,
                    atom_types=t_t,
                    frac_coords=f_t,
                    lattices_rep=l_t,
                    num_atoms=batch.num_atoms,
                    node2graph=batch.batch,
                    lattices_mat=lattices_mat_t,
                    cemb=None, guide_indicator=None,
                )
                pred = self.post_decoder_on_sample(
                    pred,
                    batch=batch, t=t_stamp,
                    anneal_lattice=anneal_lattice, anneal_coords=anneal_coords, anneal_type=anneal_type,
                    anneal_slope=anneal_slope, anneal_offset=anneal_offset,
                )
            pred_l, pred_f = pred[0], pred[1]
            if self.pred_type:
                pred_t = pred[2]
            # ========= pred each step end =========

            # ========= update each step start =========
            l_t = l_t + pred_l / N if not self.keep_lattice else l_t
            f_t = f_t + pred_f / N if not self.keep_coords else f_t
            f_t = f_t % 1.0
            if self.pred_type:
                t_t = t_t + pred_t / N
            # ========= update each step end =========

            # ========= build trajectory start =========
            if self.lattice_polar:
                lattices_mat_t = lattice_polar_build_torch(l_t)
            else:
                lattices_mat_t = l_t

            if self.pred_type:
                if self.type_encoding is None:
                    atom_types = torch.argmax(t_t, dim=-1) + 1
                else:
                    atom_types = self.type_encoding.decode_types(t_t)

            traj[t] = {
                'num_atoms': batch.num_atoms,
                'atom_types': atom_types,
                'frac_coords': f_t,
                'lattices': lattices_mat_t,
            }
            # ========= build trajectory end =========

        # stack final trajectory
        if self.pred_type:
            stack_atom_types = torch.stack([traj[i]['atom_types'] for i in range(0, N + 1)])
        else:
            stack_atom_types = batch.atom_types

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': stack_atom_types,
            'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(0, N + 1)]),
            'all_lattices': torch.stack([traj[i]['lattices'] for i in range(0, N + 1)]),
        }

        return traj[N], traj_stack

    def single_time_decoder(self, t, **kwargs):
        batch_size = kwargs["num_atoms"].shape[0]
        time_emb = self.time_embedding(t.repeat(batch_size))
        pred = self.decoder(
            t=time_emb,
            **kwargs,
        )
        return pred[0], pred[1]

    def _resign_lattice_decoder(self, t, x, **kwargs):
        return self.single_time_decoder(t=t, lattices_rep=x, **kwargs)

    def _resign_coords_decoder(self, t, x, **kwargs):
        return self.single_time_decoder(t=t, frac_coords=x, **kwargs)

    # return func with args [t, x] only
    def _partial_lattice_decoder(self, **kwargs):
        assert ("lattices_rep" not in kwargs) and ("num_atoms" in kwargs)

        def return_first(func):
            def wrapper(t, x):
                return func(t, x)[0]
            return wrapper

        f = return_first(partial(self._resign_lattice_decoder, **kwargs))
        assert getfullargspec(f).args == ["t", "x"]
        return f

    # return func with args [t, x] only
    def _partial_coords_decoder(self, **kwargs):
        assert ("frac_coords" not in kwargs) and ("num_atoms" in kwargs)

        def return_second(func):
            def wrapper(t, x):
                return func(t, x)[1]
            return wrapper

        f = return_second(partial(self._resign_coords_decoder, **kwargs))
        assert getfullargspec(f).args == ["t", "x"]
        return f

    def _fixed_odeint(
        self, batch, t_span, solver, integrate_sequence="lattice_first",
        anneal_lattice=False, anneal_coords=False,
        anneal_slope=0.0, anneal_offset=0.0,
        **kwargs,
    ):
        raise NotImplementedError("Other ode is not implemented to fit `pred_type` yet")

        assert solver.stepping_class == "fixed"

        batch_size = batch.num_graphs

        if self.lattice_polar:
            l_t = self.sample_lattice_polar(batch_size)
            if self.symmetrize_anchor:
                l_t = self.latticedecompnn.proj_k_to_spacegroup(l_t, batch.spacegroup)
            elif self.symmetrize_rotavg:
                l_t = self.latticedecompnn.proj_k_to_spacegroup(l_t, batch.spacegroup)
            lattices_mat_t = lattice_polar_build_torch(l_t)
        else:
            l_t = self.sample_lattice(batch_size)
            if self.symmetrize_anchor:
                raise NotImplementedError("symmetrize is not implemented for lattice matrix.")
            elif self.symmetrize_rotavg:
                raise NotImplementedError("symmetrize_rotavg")
            lattices_mat_t = l_t

        f_t = torch.rand([batch.num_nodes, 3]).to(self.device)
        if self.symmetrize_anchor:
            f_t_anchor = f_t[batch.anchor_index]
            f_t_anchor = torch.einsum('bij,bj->bi', batch.ops_inv[batch.anchor_index, :3, :3], f_t_anchor)
            f_t = torch.einsum('bij,bj->bi', batch.ops[:, :3, :3], f_t_anchor) + batch.ops[:, :3, 3]
        elif self.symmetrize_rotavg:
            f_t = self.symm_rotavg.symmetrize_rank1_scaled(
                scaled_forces=f_t,
                num_atoms=batch.num_atoms,
                general_ops=batch.general_ops,
                symm_map=batch.symm_map,
                num_general_ops=batch.num_general_ops,
            ) + batch.ops[:, :3, 3]

        traj = {
            0: {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': f_t.clone().detach() % 1.0,
                'lattices': lattices_mat_t.clone().detach(),
            }
        }

        """Solves IVPs with same `t_span`, using fixed-step methods"""
        _, T, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]
        pbar = tqdm(ncols=79, total=len(t_span) - 1)  # note: ignore first zero
        for steps, (_t, t) in enumerate(pairwise(t_span), 1):  # note: start from second
            dt = t - _t

            pred_l, pred_f = self.single_time_decoder(
                t=t,
                frac_coords=f_t,
                lattices_rep=l_t,
                lattices_mat=lattices_mat_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
            )

            time_emb = self.time_embedding(t.repeat(batch_size))
            pred_l, pred_f = self.post_decoder_on_sample(
                pred_l, pred_f,
                batch=batch, t=t,
                anneal_lattice=anneal_lattice, anneal_coords=anneal_coords,
                anneal_slope=anneal_slope, anneal_offset=anneal_offset,
            )

            vf_coords = self._partial_coords_decoder(
                lattices_rep=l_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
            )
            vf_lattice = self._partial_lattice_decoder(
                frac_coords=f_t,
                atom_types=batch.atom_types,
                num_atoms=batch.num_atoms,
                node2graph=batch.batch,
                lattices_mat=lattices_mat_t,
            )
            if integrate_sequence == "lattice_first":
                _, l_t, _ = solver.step(f=vf_lattice, x=l_t, t=t, dt=dt, k1=pred_l)
                _, f_t, _ = solver.step(f=vf_coords, x=f_t, t=t, dt=dt, k1=pred_f)
            elif integrate_sequence == "coords_first":
                _, f_t, _ = solver.step(f=vf_coords, x=f_t, t=t, dt=dt, k1=pred_f)
                _, l_t, _ = solver.step(f=vf_lattice, x=l_t, t=t, dt=dt, k1=pred_l)
            else:
                raise NotImplementedError("Unknown ode sequence")

            f_t: torch.Tensor = f_t % 1.0
            if self.lattice_polar:
                lattices_mat_t = lattice_polar_build_torch(l_t)
            else:
                lattices_mat_t = l_t
            traj[steps] = {
                'num_atoms': batch.num_atoms,
                'atom_types': batch.atom_types,
                'frac_coords': f_t.clone().detach(),
                'lattices': lattices_mat_t.clone().detach(),
            }

            pbar.update()

        traj_stack = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'all_frac_coords': torch.stack([t['frac_coords'] for t in traj.values()]),
            'all_lattices': torch.stack([t['lattices'] for t in traj.values()]),
        }

        return traj[list(traj)[-1]], traj_stack

    # @torch.no_grad()
    # def sample_ode(
    #     self, batch, t_span, solver, integrate_sequence="lattice_first",
    #     anneal_lattice=False, anneal_coords=False,
    #     anneal_slope=0.0, anneal_offset=0.0,
    #     **kwargs,
    # ):
    #     t_span = t_span.to(self.device)
    #     solver = str_to_solver(solver)
    #     if solver.stepping_class == "fixed":
    #         return self._fixed_odeint(
    #             batch, t_span, solver, integrate_sequence,
    #             anneal_lattice, anneal_coords,
    #             anneal_slope, anneal_offset,
    #             **kwargs,
    #         )
    #     else:
    #         raise NotImplementedError("stepping class except fixed is not accepted.")

    def training_step(self, batch, batch_idx: int, dataloader_idx=0):

        output_dict = self(batch)

        loss = output_dict['loss']
        log_dict = {
            'train_loss': loss,
            'lattice_loss': output_dict['loss_lattice'],
            'coord_loss': output_dict['loss_coord'],
            'type_loss': output_dict['loss_type'],
            'sym_lattice_loss': output_dict['loss_sym_lattice'],
            'sym_coord_loss': output_dict['loss_sym_coord'],
        }
        if self.prop_enabled:
            log_dict['prop_loss'] = output_dict['loss_property']

        if self.gradnorm is not None:
            # 用 GradNorm 动态权重组合各任务损失，并在反向传播前更新权重。
            task_losses = output_dict['loss_terms']
            loss = self.gradnorm.combine(task_losses)
            weights = self.gradnorm.update(
                task_losses, self._get_gradnorm_params(), int(self.global_step)
            )
            log_dict['train_loss'] = loss
            log_dict['train_loss_base'] = output_dict['loss']
            if weights is not None:
                log_dict.update({f'gn_w_{name}': w for name, w in weights.items()})

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx=0) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=batch.num_graphs,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': output_dict['loss_lattice'],
            f'{prefix}_coord_loss': output_dict['loss_coord'],
            f'{prefix}_type_loss': output_dict['loss_type'],
            f'{prefix}_sym_lattice_loss': output_dict['loss_sym_lattice'],
            f'{prefix}_sym_coord_loss': output_dict['loss_sym_coord'],
        }
        if self.prop_enabled:
            log_dict[f'{prefix}_prop_loss'] = output_dict['loss_property']

        return log_dict, loss

    def on_train_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_validation_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")

    def on_test_epoch_end(self) -> None:
        metrics = {"epoch": self.current_epoch}
        metrics.update({k: v.item() for k, v in self.trainer.logged_metrics.items()})
        metriclogger.info(f"{metrics}")
