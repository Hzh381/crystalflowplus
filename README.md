# CrystalFlow: A Flow-Based Generative Model for Crystalline Materials

Implementation codes for CrystalFlow: A Flow-Based Generative Model for Crystalline Materials.

For MindSpore implementation please refer to [here](https://gitee.com/mindspore/mindscience/tree/master/MindChemistry/applications/crystalflow).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ixsluo/CrystalFlow/blob/main/LICENSE)   [**[Paper]**](https://arxiv.org/abs/2412.11693)

### Dependencies and Setup

Note: updated `python==3.11` and `pytorch>=2.0.0` and `hydra>=1.3`

```bash
# sudo apt-get install gfortran libfftw3-dev pkg-config
conda create -n crystalflow python=3.11.9
conda activate crystalflow
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install lightning==2.3.2 finetuning_scheduler
pip install hydra-core omegaconf python-dotenv wandb swanlab rich
pip install p_tqdm pymatgen pyxtal smact matminer einops chemparse torchdyn
pip install -e .
mkdir -p log hydra
```

Rename the `.env.template` file into `.env` and specify the following variables.

```
PROJECT_ROOT: the absolute path of this repo
HYDRA_JOBS: the absolute path to save hydra outputs
WANDB_DIR: the absolute path to save wabdb outputs
```

example:
```bash
# vi .env
export PROJECT_ROOT="/home/<YOURHOME>/CrystalFlow"
export   HYDRA_JOBS="/home/<YOURHOME>/CrystalFlow/hydra"
export    WANDB_DIR="/home/<YOURHOME>/CrystalFlow/log"
```

### Datasets

MP20 and MPTS52 datasets are provided in `dataset/mp_20` and `dataset/mpts_52`.

### Training

test passed on version `v1.0.0-alpha.1`.

Pretrained checkpoints are provided in [Releases](https://github.com/ixsluo/CrystalFlow/releases).

#### For CSP task on MP-20 dataset without conditioning

```bash
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
data=mp_20 data.train_max_epochs=3000 \
model=flow_polar \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=CSP-mp20 \
      > CSP-mp20.log 2>&1 &
```

The checkpoints and other files will be in `hydra/singlerun/CSP-mp20`.

Run time on RTX-4090 is about 24 hours.

#### For CSP task on MP-CALYPSO-60 dataset with pressure conditioning

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
data=calypso_60 data.train_max_epochs=2000 \
data.datamodule.batch_size.train=64 \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
model=flow_polar \
+model.guide_threshold=-1 \
+model.from_cubic=false \
+model.lattice_polar_sigma=0.1 \
model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=CSP-mpcalypso60-pressure \
      > CSP-mpcalypso60-pressure.log 2>&1 &
```

#### For DNG task on MP-20 without conditioning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
data=mp_20 data.train_max_epochs=3000 \
model=flow_polar_w_type \
+model.type_encoding=table \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_type=10 model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=DNG-mp20 \
      > DNG-mp20.log 2>&1 &
```

#### For DNG task on MP-20 with formation energy conditioning

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 HYDRA_FULL_ERROR=1 nohup python diffcsp/run.py \
train.pl_trainer.devices=4 \
+train.pl_trainer.strategy=ddp_find_unused_parameters_true \
data=mp_20_chgnet data.train_max_epochs=3000 \
model=flow_polar_w_type \
+model.type_encoding=table \
+model.guide_threshold=-1 \
optim.optimizer.lr=1e-3 \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
+model.lattice_polar_sigma=0.1 \
model.cost_type=10 model.cost_coord=10 model.cost_lattice=1 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=online \
logging.wandb.project=crystalflow-gridtest \
expname=DNG-mp20-Eform \
      > DNG-mp20-Eform.log 2>&1 &
```


### CrystalFlow+ 改进：条件引导 / 多任务平衡 / 物性预测

本仓库在原始 CrystalFlow 主干（CNF + 条件流匹配、晶格参数与分数坐标联合生成、周期等变 GNN、原 ODE 采样）之上，新增三项可独立开关的改进，统一通过 `model=flow_plus` 启用。原有的 `model=flow` / `flow_polar` / `flow_polar_w_type` 等配置与旧 checkpoint 的加载、训练和采样行为保持不变。

1. **Classifier-Free Guidance（CFG，条件引导）**
   - 训练时以 `cond_drop_prob` 的概率丢弃物性条件，使同一网络同时学习条件流场 `v(x|y)` 与无条件流场 `v(x)`；
   - 采样时 `v_cfg = v_uncond + s(t) · (v_cond − v_uncond)`；
   - 支持时间自适应引导强度 `s(t) = s_min + (s_max − s_min) · u^γ`：`u=t`（`ascending`）或 `u=1−t`（`descending`），便于实验比较不同生成阶段的约束强弱；
   - 对应代码：`diffcsp/pl_modules/flow.py` 中的 `cond_drop_prob` 与 `sample()`。

2. **GradNorm（多任务动态损失平衡）**
   - 在共享 GNN 的最后一层（默认 `decoder.csp_layer_5`）计算晶格流、坐标流（以及物性头）的梯度范数，按相对训练速率自动调整各项损失权重；
   - 训练日志新增 `gn_w_*`（各任务权重）与 `train_loss_base`（未加权的基准损失）；
   - 对应代码：`diffcsp/pl_modules/gradnorm.py` 与 `flow.py` 的 `training_step()`。

3. **物性预测头（辅助监督 + 生成筛选）**
   - 在 CSPNet 的图级特征上增加 MLP 预测头；
   - 默认的 `clean` 监督方式在真实（t=1）结构上、且不输入条件向量地进行预测，避免把噪声中间态或目标条件泄漏进预测；
   - 另提供更轻量的 `time_gated` 模式（只对 `t ≥ t_min` 的样本、按 `t^power` 加权监督）；
   - 提供 `predict_property()` 与 `scripts/property_screen.py`，可从同一组成的多个候选结构中选出最接近目标物性的结果。

#### 配置开关

`conf/model/flow_plus.yaml` 中的主要参数：

| 键 | 含义 | 默认值 |
| --- | --- | --- |
| `cond_drop_prob` | 训练时丢弃条件的概率（CFG） | `0.15` |
| `property_head.enabled` | 是否启用物性预测头 | `true` |
| `property_head.key` | 监督的物性键名；`null` 时取 `data.conditions[0]` | `null` |
| `property_head.supervision` | `clean`（真实结构）或 `time_gated`（时间门控） | `clean` |
| `property_head.clean_freq` | 每 N 个训练步执行一次干净结构前向 | `1` |
| `gradnorm.enabled` | 是否启用 GradNorm | `true` |
| `gradnorm.shared_module` | 测量梯度范数的共享模块 | `decoder.csp_layer_5` |

#### 训练

完整启用三项改进：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python diffcsp/run.py \
data=mp_20_ehull model=flow_plus \
optim.optimizer.lr=1e-3 \
expname=CrystalFlowPlus-mp20-ehull
```

只启用其中一部分（示例）：

```bash
# 只启用 CFG
python diffcsp/run.py data=mp_20_ehull model=flow_plus \
model.gradnorm.enabled=false model.property_head.enabled=false \
expname=CrystalFlowPlus-cfg-only

# 物性头改用轻量时间门控监督
python diffcsp/run.py data=mp_20_ehull model=flow_plus \
model.property_head.supervision=time_gated \
expname=CrystalFlowPlus-time-gated
```

对应的模型文件、hparams 与 checkpoint 位于 `hydra/singlerun/<expname>`。

#### 条件采样（CFG）

```bash
python scripts/generation.py -m hydra/singlerun/CrystalFlowPlus-mp20-ehull \
-N 100 --dataset mp_20 \
--guide-factor 1.0 --guide-s-min 0.5 --guide-s-max 2.5 --guide-gamma 1.0 \
--guide-time-dir ascending \
--conditions "e_above_hull=0.0" --label cfg
```

- `--guide-factor`：不启用时间调度时的常数引导强度；
- `--guide-s-min` / `--guide-s-max` / `--guide-gamma`：时间自适应强度曲线；
- `--guide-time-dir`：`ascending` 表示引导随 t（噪声→数据）增强，`descending` 相反；
- 未用 CFG 训练的模型不能传入引导参数。

#### 生成结果的物性筛选

```bash
python scripts/property_screen.py \
-m hydra/singlerun/CrystalFlowPlus-mp20-ehull \
-g hydra/singlerun/CrystalFlowPlus-mp20-ehull/eval_gen_cfg.pt \
-k 0.0 --top-k 20 --write-cif
```

输出 `screened.pt`（含预测值、误差、候选索引与结构），`--write-cif` 时同时写出选中的 CIF。

#### 说明与限制

- `supervision: clean` 每个训练步会额外执行一次 decoder 前向，训练耗时与显存约增加；可用 `clean_freq` 降频或切换为 `time_gated`。
- `property_screen.py` 目前最适用于 ab-init（无空间群投影）模型；对称生成模型需要额外提供 `spacegroup` 才能复用 `predict_property()`。
- 旧配置名 `guide_threshold` 仍被识别为 `cond_drop_prob` 的别名；原有 `model=flow*` 配置的训练/采样行为保持不变。

### Evaluation

#### CSP generation

One sample

```bash
python /path/to/scripts/evaluate.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --anneal_coords --anneal_slope 5 --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

Multiple samples

```bash
python /path/to/scripts/evaluate.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --num_evals 20 --anneal_coords --anneal_slope 5 --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks csp --gt_file data/<dataset>/test.csv --multi_eval --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

##### extract CSP generation

```bash
python /path/to/scripts/extract_gen.py eval_gen_<label>.pt --task eval
# will save to eval_gen_<label>.dir
```

#### DNG generation

```bash
python /path/to/scripts/generation.py --model_path <model_path> --ode-int-steps 100 --dataset <dataset> --label <label>
python /path/to/scripts/compute_metrics.py --root_path <model_path> --tasks gen --gt_file data/<dataset>/test.csv --label <previous-label>
```

Results will be saved to the same dir as `model_path`.

##### extract DNG generation

```bash
python /path/to/scripts/extract_gen.py eval_gen_<label>.pt --task gen
# will save to eval_gen_<label>.dir
```


#### Sample from arbitrary composition

```bash
python /path/to/scripts/sample.py --model_path <model_path> --ode-int-steps 100 --save_path <save_path> --formula <formula> --num_evals <num_evals> --anneal_coords --anneal_slope 5
# will save to the <save_path>
```

### Acknowledgments

The main framework of this codebase is build upon [DiffCSP](https://github.com/txie-93/cdvae). For the datasets, Perov-5, Carbon-24 and MP-20 are from [CDVAE](https://github.com/txie-93/cdvae), and MPTS-52 is from [DiffCSP](https://github.com/txie-93/cdvae) (originaly from [codebase](https://github.com/sparks-baird/mp-time-split)).

### Citation

Please consider citing our work if you find it helpful:
```
X. Luo et al., CrystalFlow: A Flow-Based Generative Model for Crystalline Materials, arXiv 2412.11693
```

### Contact

If you have any questions, feel free to reach us at:

Xiaoshan Luo: [luoxs21@mails.jlu.edu.cn](mailto:luoxs21@mails.jlu.edu.cn)
