# SIMI_rebuild

对原项目 `../model/SIMI` 中 `model_new.py`、`train_new.py`、`dataloader_new.py`
和 `Myloss.py` 的模块化研究框架。原项目保留，研究模块在此目录中只有一份实现。
当前默认增强主干已修正为 DEA，并修正首级 B 预测头；历史源码通过显式模式复现。

## 保持的行为

- 当前默认结构：24 个位平面、两层位平面平滑、CBAM、三层融合后卷积、
  27→32 通道投影、七个串联 DEA 块、七组独立 A/B 预测头和七次原始递推，
  以及默认去噪分支。
- DEA 增强参数 **377,732**，去噪参数 **110,312**，完整模型 **488,044**。
  `legacy_cnn` 增强参数 **105,907**，完整模型 **216,219**。
- 位平面、编码器、曲线公式、损失公式和去噪结构保持原样；DEA 的参数结构与旧 CNN 不同。
- 位平面顺序为 `R7…R0, G7…G0, B7…B0`，值为 0/1。
- 训练先将 uint8 RGB 按 LANCZOS 缩放至配置尺寸（默认 512×512），再归一化；
  训练器在训练设备上通过 `round(RGB * 255)` 还原 uint8 并批量拆位，
  与原 uint8 拆位结果一致。数据集默认接口仍支持直接返回 RGB 和位平面。
- 默认 joint 在同一次前向产生两个输出；先更新增强网络，再以该次增强结果作为 detached
  目标更新去噪网络。增强结果及末级特征送入去噪前均 detach。
- 两个 AdamW，学习率 1e-5、weight decay 1e-4、batch size 8；
  梯度裁剪 0.1；两个 StepLR 随对应优化器的成功更新推进，每 250 步乘 0.99。
- 损失公式、权重、归约方式与原训练脚本一致；去噪 SSIM 显式使用归一化图像范围 1.0。
- 默认稳定排序文件列表，由独立采样器打乱；统一设置 Python、NumPy、PyTorch、CUDA
  和加载 worker 的随机种子，默认 `runtime.seed: 1143`、`runtime.deterministic: true`。
  没有新增数据增强或验证集训练反馈；CUDA 训练可配置 AMP。

默认首级 B 使用 `e_conv1_b`，所有七级 A/B 头均独立参与预测。
去噪中的未使用 BN 仍保留，以保持参数数量和权重兼容。
去噪训练 SSIM 的 `data_range` 已修正为 **1.0**；历史实验可显式配置为 255.0，
该兼容值不会自动启用。论文与旧代码间的曲线公式等其它差异保持原先处理方式。

DEA 按 [DEA-Net 官方训练实现](https://github.com/cecret3350/DEA-Net/blob/main/code/model/modules/deablock_train.py)
组装：DEConv → ReLU → 第一次残差 → 普通卷积 → CGA → 第二次残差。
DEConv 使用中心、水平、垂直、角度差分以及普通卷积五分支；CGA 使用通道、空间和
像素注意力，逐通道交错组合像素输入。支持 CPU/CUDA 和随权重的数据类型，不写死 CUDA。
块外不添加 ReLU。模型输入输出维度保持不变。
DEA 块内部保留官方实现的默认构造初始化；原位平面编码器、输入投影和 A/B 头
使用原初始化函数，历史模式仍保持全部旧初始化顺序。

`train.activation_checkpointing: true` 在 DEA 训练时重算块内激活，以保留
RTX 3090 上 batch 8、512×512 的配置。该策略不添加参数或改变曲线、loss，
推理与冻结增强分支不重算；可设为 `false`，显存占用会增加。

当前 PyTorch 的 CUDA 自适应最大池化反向不支持严格确定性。CBAM 在严格 CUDA
模式下使用等价的全局最大值归约，输出、有限输入的梯度和注册参数不变；
该模式遇到非有限特征会明确报告错误。关闭确定性时沿用原池化实现。

论文明确 D=7，但未完整说明隐藏宽度、输入投影和融合卷积配置。
这里沿用旧实现的隐藏宽度 32、融合卷积及去噪分支，增加必要的 27→32 残差通道投影。
参数量按未经推理重参数化的训练模型统计；未声称等于论文的 **0.122M** 或严格复现全部论文细节。

## 目录职责

| 路径 | 职责 |
|---|---|
| `configs/default.yaml` | 当前训练默认值、路径和 loss 权重 |
| `datasets/bitplane.py` | 唯一位平面分解实现，支持 CHW/BCHW 和 CPU/CUDA |
| `datasets/lowlight_dataset.py` | 场景目录中的无配对训练图像 |
| `datasets/sampler.py` | 可恢复的样本顺序及已完成 batch 游标 |
| `models/bit_encoder.py` | 24 通道平滑、RGB 融合和 CBAM |
| `models/enhancement.py` | 七级曲线预测及原递推式 |
| `models/denoising.py` | 独立默认去噪网络 |
| `models/attention.py`, `blocks.py` | 共用注意力、卷积激活和初始化 |
| `models/simi.py` | 两分支编排、研究接口及权重兼容 |
| `losses/unsupervised_loss.py` | 增强与去噪损失 |
| `trainers/trainer.py` | 训练更新、调度和检查点 |
| `trainers/validator.py` | 六个测试集独立评价，共享官方 LPIPS |
| `utils/logging.py` | 终端和文件实时日志，统一输出目录 |
| `inference/test.py` | 单图/目录推理及可选配对指标 |
| `utils/` | 图像与指标、统一配置路径、随机状态及 CUDA 同步计时 |

没有复制旧版 RGB baseline 或可视化实验。DEA 共用组件集中在 `blocks.py`、`attention.py`；
历史代码仍可在原 `../model/SIMI` 中查阅。

## 环境和训练

继续使用已经配置的 `ild-net`。配置使用 PyYAML，验证使用已安装的 `lpips`；
不需要更换 PyTorch/CUDA。官方 AlexNet 预训练特征和 LPIPS v0.1 权重已在本机可用。

```bash
conda activate ild-net
cd /home/dingyujie/GY/SIMI_rebuild
python train.py --config configs/default.yaml
```

默认启用 CUDA FP16 AMP；CPU 自动沿用 FP32。模型参数和 AdamW 状态仍为 FP32，
卷积特征与曲线预测使用 autocast；七次曲线递推、全部损失和验证指标保留 FP32。
AMP 会带来浮点数值差异，网络参数数量、曲线公式和损失权重保持原样。

```yaml
amp:
  enabled: true
  dtype: float16
  init_scale: 64.0
  growth_factor: 2.0
  backoff_factor: 0.5
  growth_interval: 2000
```

也可以用命令行选择精度：

```bash
python -u train.py --config configs/default.yaml --amp --amp-dtype float16
python -u train.py --config configs/default.yaml --amp --amp-dtype bfloat16
python -u train.py --config configs/default.yaml --no-amp
```

RTX 3090 支持 BF16；其它设备会在启动时检查兼容性。FP16 使用动态 GradScaler，
BF16 不需要梯度缩放。按照 [PyTorch AMP 示例](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html)，
先还原梯度再裁剪，两个分支各自判断是否成功更新，共享缩放器每个 iter 只更新一次。
溢出分支跳过 AdamW 及其 scheduler，另一分支可正常更新；iter 仍记录已处理的 batch。
日志增加精度、loss scale 和两个分支的累计跳过次数。初始 scale 64 适应 SIMI 较大
损失权重，遇到溢出会自动降低，随后按 growth_interval 调整。

从随机初始化开始训练，默认使用 FP16 AMP；日志自动保存，无需外部 `tee`：

```bash
python -u train.py --config configs/default.yaml --amp --output-dir outputs
```

默认数据路径是 `../model/dataset/SICE/SIMI/train`，当前包含 3021 张图片，格式为
`root/scene/*.jpg`。相对 YAML 路径按项目目录解析；命令行覆盖路径按当前工作目录解析。
默认 joint 训练 500 epochs，batch size 8、`drop_last: false`，每轮 378 iter，
共 189,000 iter。每个 epoch 完成后依次评价全部六个测试集，合计 515 对：

| 数据集目录名 | 配对数 | 图像尺寸 |
|---|---:|---|
| `LOLv1` | 15 | 600×400 |
| `LOLv2real` | 100 | 600×400 |
| `LOLv2syn` | 100 | 384×384 |
| `LSRWhuawei` | 30 | 960×720 |
| `LSRWNikon` | 20 | 960×640 |
| `SICE` | 250 | 512×512 |

每个数据集独立保存三个最佳模型，输出结构如下：

```text
outputs/
├── logs/
│   ├── train.log
│   └── validation_metrics.csv
└── weights/
    ├── latest.pth
    ├── LOLv1/
    ├── LOLv2real/
    ├── LOLv2syn/
    ├── LSRWhuawei/
    ├── LSRWNikon/
    └── SICE/
        ├── best_psnr.pth
        ├── best_ssim.pth
        ├── best_lpips.pth
        └── best_models_metrics.txt
```

六个数据集文件夹都包含上面四个文件，各自的更新规则为：

| 文件 | 更新条件 |
|---|---|
| `best_psnr.pth` | 验证集平均 PSNR 严格提高 |
| `best_ssim.pth` | 验证集平均 SSIM 严格提高 |
| `best_lpips.pth` | 验证集平均 LPIPS 严格降低 |
| `best_models_metrics.txt` | 每轮刷新三个最佳模型的完整指标、epoch 和 global step |
| `outputs/weights/latest.pth` | 六集验证后统一更新；达到 `--max-steps` 停点也更新 |

各集三个最优文件独立选择，分数相同时保留较早的权重。全部保存完整续训状态。
摘要中的 PSNR/SSIM/LPIPS 来自该最佳模型所在轮次，不能把三项各自的最优数值
拼成一组。展示 epoch 从 1 开始；内部检查点仍用从 0 开始的轮次。
`train.log` 自动追加终端输出、警告和异常，保留每 100 iter 的 loss 和 ETA。
`validation_metrics.csv` 保存每轮每集完整指标，续训按检查点历史重建以避免重复。
不再生成 `EpochN.pth`、`StepN.pth` 或 `last.pth`，旧文件不会自动删除。

`validation.datasets` 配置每个测试集名称及其 `low_dir`、`high_dir`，共享
顶层 `validation.lpips_net: alex`。同名严格配对与尺寸校验在训练开始前完成。
可用 `--output-dir outputs/experiment_name` 将一整次实验的日志和权重放入子目录；
`--snapshots-folder` 若单独覆盖，必须位于当前 `output.root` 内。
SICE 使用 Part II 的 250 对
同名 512×512 PNG（125 个场景各前两张曝光及对应 Label）；Part I 的 3021 张
训练图像与它们不重叠。此目录用于轮末模型选择，参考图像不进入训练损失。
Part II 的排序及 resize 处理约定见 `../model/dataset/SICE/SIMI/derived.json`。
验证保持尺寸，使用未量化、未截断的浮点输出，并逐图片计算后平均。
PSNR/SSIM 使用范围 1.0；LPIPS 使用官方 AlexNet v0.1，输入线性变换为
`2 * image - 1`。LPIPS 是距离指标，越低越好。
增强模式只验证增强输出；joint 和 denoising 模式验证最终去噪输出。
六集共享一个冻结的 LPIPS，不增加 SIMI 参数，也不参与优化器；
验证后恢复所有训练模式和随机状态。六集全部成功评价后才整体提交最佳指标历史。

默认每 100 iter 输出当前 batch 的各项 loss、epoch/batch/总 iter 进度、
两个优化器学习率及 ETA。ETA 使用已观测的训练和轮末验证平均耗时估算，
首轮验证完成前只具备训练耗时估计；暂停时间不计入，完整续训恢复累计计时。
例如日志格式为：

```text
[joint] Epoch 1/500 Batch 100/378 Iter 100/189000 lr_light=1e-05 lr_denoise=1e-05 loss_light: ... ETA=2d 03:04:05
[validation:LOLv1] Epoch 1/500 images=15 PSNR=... SSIM=... LPIPS=... best_updated=psnr,ssim,lpips ETA=...
[validation:SICE] Epoch 1/500 images=250 PSNR=... SSIM=... LPIPS=... best_updated=psnr,ssim,lpips ETA=...
```

可覆盖模式、设备、数据、轮数、batch 和 worker 数：

```bash
python train.py --mode joint --device cuda:0 --data-root /path/to/train \
  --epochs 500 --batch-size 8 --num-workers 4
```

完整断点续训：

```bash
python train.py --resume outputs/weights/latest.pth
```

新版检查点保存模型、两个优化器、两个 scheduler、epoch、global step、已完成的
batch 游标、数据清单、采样排列、loader generator 和全部随机状态，
同时保存最佳分数/轮次、逐轮验证历史、验证集及指标定义指纹、ETA 计时状态，
以及 GradScaler 和累计溢出跳过次数。
采样器只在更新成功后确认游标，worker 预取不会造成恢复时跳过样本。
写入采用临时文件替换，`torch.load(..., weights_only=True)` 可读取完整状态。
`epochs` 表示目标总轮数；续训不会重新执行已完成的 batch。
`--max-steps N` 可在总更新数达到 N 时安全保存并停止。

`train.resume_mode` 明确区分恢复用途：

| 模式 | 行为 |
|---|---|
| `full`（默认） | 恢复完整训练状态；数值设置、训练模式、设备和环境必须匹配 |
| `weights` | 只载入模型；重新创建优化器、scheduler 和阶段进度，使用当前 YAML 超参数 |
| `legacy` | 显式载入模型和优化器作热启动；epoch、scheduler 和随机状态从新进度开始 |

`--pretrain PATH --resume-mode weights` 用于切换阶段。
原始旧检查点缺少精确训练进度，必须显式选 `weights` 或 `legacy`。
此前 v2 完整检查点可恢复原训练进度；因没有验证历史，会从下一次轮末验证重新
建立最优分数及 ETA 计时。v3 检查点完整恢复验证历史，避免把较差模型覆盖为最佳。
多集验证使用 v5 检查点，保存每集最佳三元组和独立历史；旧单集配置仍保存 v4。
v3/v4 的单集历史可按唯一匹配的验证指纹迁移至六集中对应数据集，其余集从下轮
首次验证建立最佳记录，同时迁移真正的历史最佳文件。指纹不匹配时明确拒绝恢复。
v2/v3 FP32 检查点可保留原进度切换至 AMP，缩放器从当前配置初始化。
v4/v5 精确续训要求有效 AMP 配置一致；改变精度应开启新实验。
`resume_optimizers: false` 仍兼容仅载权重的旧配置写法。
`legacy` 也允许用新版检查点显式热启动，不代表恢复原训练进度。
DEA 保存当前模块键名；`legacy_cnn` 保存旧 `model_new` 权重键名，并加入完整状态。
裸 state_dict、旧训练字典及新版字典均可用于匹配主干的推理和仅载权重。

数据恢复会校验文件相对路径、大小和修改时间；目录可迁移，但文件属性必须匹配。
验证集还会校验配对指纹和指标定义，改变验证集应使用 `weights` 开始新实验。
完整续训会检查三个历史最佳文件与恢复的指标历史一致；更换输出目录时自动迁移
对应最佳文件。缺失历史权重或从旧停点回退而目录已有不匹配的后续最佳文件时，
会明确拒绝，避免把当前权重当成历史最优。可提供匹配文件，或以 `weights` 开始新实验。
当前确定性数据读取没有随机增强，已验证多 worker 中途恢复。
未来加入随机数据增强时，需要同步扩展 worker 状态恢复。
外部直接调用 `train_step` 保存的检查点可继续直接更新；缺少数据游标时，
`fit` 完整恢复会明确拒绝，避免错误声称可以恢复样本进度。

默认 `enhancement_backbone: dea`。旧 CNN 权重不能直接加载到 DEA，不能用
`strict=False` 跳过结构差异；旧实验应显式设置 `enhancement_backbone: legacy_cnn`，
或使用训练、推理的 `--enhancement-backbone legacy_cnn`。
历史模式包含旧首级 B 共享，仅用于旧检查点兼容和逐项回归，不包含本轮算法修正。

## 三种训练模式

在完整 `configs/default.yaml` 中设置 `train.mode`，其余训练配置保持显式填写：

| `train.mode` | 更新的分支 | 前向与损失 |
|---|---|---|
| `enhancement` | 增强网络 | RGB → 位平面提取 → 增强；局部颜色、全局颜色、曝光、A/B 平滑损失 |
| `denoising` | 去噪网络 | RGB → 位平面提取 → 冻结增强 → 去噪；原 SSIM、梯度、输出平滑损失 |
| `joint` | 两个分支 | 一次完整前向；原增强 loss 更新增强，再以 detached 目标用原去噪 loss 更新去噪 |

冻结分支同时关闭参数梯度并使用评估模式，其优化器和 scheduler 不更新。
Stage 2 中增强前向使用 `torch.no_grad()`，去噪前向继续构建梯度图。
`joint` 保留原始两分支的梯度隔离与更新顺序；SSIM 范围之外的损失运算保持原样。

先将 `train.mode` 设为 `enhancement` 完成 Stage 1；进入 Stage 2 时，在完整配置中
修改以下字段，将路径换成 Stage 1 实际生成的检查点：

```yaml
output:
  root: outputs/stage2
train:
  mode: denoising
  load_pretrain: true
  pretrain_dir: outputs/stage1/weights/SICE/best_psnr.pth
  resume_optimizers: false
  resume_mode: weights
  snapshots_folder: outputs/stage2/weights
```

以上仅列出需要修改的字段，应合并到完整配置。Stage 1 可用
`--output-dir outputs/stage1`；Stage 2 执行 `python train.py --config configs/default.yaml`。
训练不会自动在阶段间切换，配置中的 `epochs` 控制当前阶段的轮数。

学习率、AdamW 参数、batch size、epochs、梯度裁剪、scheduler、数据路径、
图像尺寸、worker 数、shuffle、pin_memory、drop_last 以及全部 loss 权重均从 YAML
读取，不使用代码中的数值回退。缺少必需配置或模式名称错误会在训练前报告。
`Trainer.train_step(rgb)` 在设备上提取位平面；`Trainer.train_step(rgb, planes)`
兼容显式位平面输入，不会重新提取或覆盖该输入。`fit()` 读取 RGB-only batch，
执行与当前模式对应的管线，并保存完整训练检查点。

## RGB 输入和研究接口

完整 YAML 使用以下研究选项和主干配置：

```yaml
bit_mode: binary
enhancement_guidance: none
enhancement_backbone: dea
```

| 配置 | 候选值 | 当前实现 |
|---|---|---|
| `bit_mode` | `binary`、`gray`、`learnable`、`reliable` | 仅 `binary` |
| `enhancement_guidance` | `none`、`snr`、`reliability` | 仅 `none` |
| `enhancement_backbone` | `dea`、`legacy_cnn` | 默认 DEA；历史 CNN 兼容 |

训练配置要求显式填写这三个字段。位平面和引导的未来候选表示预留接口；未提供对应研究模块时，
工厂会抛出 `NotImplementedError`，不会自动退回 binary 或忽略引导模式。
非法名称会报告 `ValueError`。没有实现 Gray-code、可学习位平面、可靠性位平面、
SNR 估计或可靠性引导算法。

`datasets.build_bitplane_extractor` 与 `models.build_enhancement_network` 集中管理
默认实现和显式模块注入。SIMI 只注册一个提取器实例，训练和推理均引用该实例；
输入输出、七级曲线与 loss 保持原样。
推理可通过 `--config configs/default.yaml` 读取这些选项，
训练与推理支持 `--bit-mode`、`--enhancement-guidance`、`--enhancement-backbone` 覆盖。
推理的图片、检查点、输出和阶段可在 YAML 的 `inference` 节设置，
设备与训练共用 `runtime.device`；显式命令行参数优先。

输入浮点 RGB 为 `[B,3,H,W]`、范围 `[0,1]`。显式传位平面时，其形状为
`[B,24,H,W]`。不传位平面时，模型将 RGB×255 四舍五入至 uint8 后调用同一分解函数。
需要逐字节复现数据加载时，应直接从原 uint8 图拆位。

```python
import torch
from models import SIMI

model = SIMI().cuda().eval()
rgb = torch.rand(1, 3, 512, 512, device="cuda")
with torch.inference_mode():
    final_rgb = model.enhance(rgb)  # RGB -> 最终 RGB，保留默认去噪阶段
    enhanced, curve_a, curve_b, denoised = model(rgb)  # 原四项输出接口
    features = model.extract_features(rgb)

assert features["rgb"].shape[1] == 3
assert features["raw_bitplanes"].shape[1] == 24
assert features["bit_plane_features"].shape[1] == 24
assert features["fused_features"].shape[1] == 27
```

`extract_features` 返回保留梯度的原 RGB、原始位平面、平滑后 24 通道特征和
27 通道融合特征，不增加参数。`model(rgb, bitplanes)` 仍返回原四项 tuple。

`model.load_state_dict(state)` 严格加载与所选主干相匹配的权重；
只有 `SIMI(enhancement_backbone="legacy_cnn")` 支持完整旧 `model_new` 格式，
并提供 `legacy_state_dict()` 导出。DEA 导出使用 `state_dict()`。
缺失键、多余键、形状或主干不一致会报错，不静默加载部分权重。

后续实现 `ReliableBitPlaneExtractor` 时继承 `BitPlaneExtractor`，仍返回
`[B,24,H,W]` 张量。引导图独立通过 `extra_guidance` 传递，提取器不改变返回格式。
增强扩展可以覆盖 `CurveEnhancementNetwork._prepare_features` 或 `forward_with_aux`；
集成完整 SIMI 时使用 `EnhancementNetwork` 兼容子类，保持训练所需的四项输出。
以下仅展示后续模块的接入方式，变量应由未来研究实现提供：

```python
model = SIMI(
    bit_mode="reliable",
    enhancement_guidance="reliability",
    bitplane_extractor=research_extractor,
    enhancement_network=research_enhancement,
)
enhanced, curve_a, curve_b, denoised = model(rgb, extra_guidance=reliability_map)
```

`Trainer(config, bitplane_extractor=..., enhancement_network=...)` 支持同样的注入，
`train_step(rgb, planes=None, extra_guidance=None)` 在三种训练模式下透传引导图。
默认数据集不生成引导图；传入默认增强网络的非空引导仍会报错。
当前训练器明确拒绝带参数的位平面提取器，未来 learnable 模块的优化器归属、
冻结策略和扩展检查点需随模块一并实现。旧检查点兼容使用显式历史主干。

## 独立曲线增强接口

`CurveEnhancementNetwork.forward(image, bit_feature, extra_guidance=None)` 接收
`[B,3,H,W]` RGB 和 `[B,24,H,W]` 位平面，返回 `[B,3,H,W]` 增强 RGB。
两个输入应为相同设备、相同浮点类型。这里的 `bit_feature` 是尚未经过平滑的
原始位平面；网络内部继续执行原有平滑、CBAM 和融合，不能传入已经平滑的
`extract_features()["bit_plane_features"]`。

```python
from datasets import BinaryBitPlaneExtractor
from models import CurveEnhancementNetwork

curve_model = CurveEnhancementNetwork().cuda().eval()
bit_feature = BinaryBitPlaneExtractor()(rgb)
with torch.inference_mode():
    enhanced = curve_model(rgb, bit_feature, extra_guidance=None)
```

仍先预测七组 `A_i = tanh(...)`、`B_i = sigmoid(...) * 0.5 + 0.5`，再按原公式
执行七次 RGB 递推。`forward_with_aux` 返回增强图、两组 21 通道曲线参数和
末级 32 通道特征，供原损失和去噪分支使用。现有 `EnhancementNetwork` 继承此实现，
仅适配原四项返回值。DEA 与历史模式分别有 **377,732**、**105,907** 个参数，
相同主干配置下可直接加载
`curve_model.load_state_dict(model.light_net.state_dict())`。

`extra_guidance` 为可靠性图、SNR 图和自适应曲线控制保留入口，默认 `None`
执行原路径；当前传入非空值会抛出 `NotImplementedError`。
未来可覆盖 `_prepare_features` 接入特征引导，或覆盖 `forward_with_aux` 调用
`curve_network.predict_curves` 与 `curve_network.apply_curves` 扩展曲线控制。
默认实现未增加引导算法。

## 推理

```bash
python inference/test.py \
  --checkpoint outputs/weights/SICE/best_psnr.pth \
  --input /path/to/low.png \
  --output results/enhanced.png \
  --device cuda:0
```

默认输出去噪后的 RGB；`--stage enhancement` 仅运行增强分支并输出增强结果，
跳过去噪计算。模型参数、检查点格式、训练流程和输出图像数值保持不变。
目录输入只读取该目录中的图片，输出到指定目录。保持原始图像尺寸，模型只加载一次。
可用 `--gt /path/to/high.png` 或同名 GT 目录计算原 PSNR/SSIM，逐图聚合；
指标使用未量化输出、data_range=1.0，与原测试计算方式一致。保存图像沿用
torchvision 的 clamp/round 行为。

默认启用模型前向计时，每张图先按 `inference.warmup: 3` 预热，再测量一次所选
阶段的前向；CUDA 在计时前后同步所选设备。加载、图像读写、位平面拆分和
质量指标不计入该时间。输出逐图耗时和平均耗时；预热不重复保存图像或计入指标。
可使用 `--no-benchmark` 关闭计时，或 `--warmup 0` 关闭预热。

```yaml
runtime:
  device: cuda:0
inference:
  checkpoint: outputs/weights/latest.pth
  input: /path/to/low.png
  output: results/enhanced.png
  gt: ""
  stage: enhancement
  benchmark: true
  warmup: 3
```

以上字段应合并到完整配置；之后执行 `python inference/test.py --config configs/default.yaml`。

## 重构验证

```bash
python -B -m unittest discover -s tests -v
```

回归测试自动查找 `../SIMI` 或 `../model/SIMI` 的原源码，实现位平面穷举、
参数数量、初始化与注册顺序、前向输出、参数梯度、损失及其梯度、真实原训练
函数的一次更新、检查点兼容验证。独立曲线接口另覆盖七级预测与递推、引导扩展
入口以及 CPU/CUDA 输出和梯度严格等价。CUDA 对照测试局部使用确定性卷积，
退出后恢复原设置；历史 CUDA 对照局部关闭原源码不支持的严格确定性。
找不到原源码时，相应对照测试会明确跳过。
训练器测试覆盖三种模式、冻结参数与缓冲区、梯度隔离、优化器与调度器顺序、
设备内拆位入口、三种检查点恢复策略、完整 YAML 生效以及真实 JPG 加载与保存。
实验接口测试覆盖模式枚举与工厂、未实现模式拒绝、显式模块注入、提取器唯一归属、
三种训练模式的引导图透传，以及实际单图推理中的 YAML 与命令行优先级。

新增 DEA 测试使用独立差分核、注意力和双残差公式验证输出与梯度，检查七个块、
十四个独立预测头、不同结构的检查点拒绝混用，以及三种训练模式的保存、恢复与推理。
新增 SSIM 范围、统一推理配置与同步计时、确定性注意力及完整恢复测试；
恢复测试对照三种训练模式和双 worker 的模型、优化器、scheduler、样本顺序及随机状态，
并覆盖阶段切换、数据变化拒绝、正常结束的 latest.pth 和外部更新缺少游标的拒绝。
轮末验证测试进一步覆盖官方预训练 LPIPS、三指标独立选优、平分不覆盖、
验证历史和 ETA 恢复、v2 迁移、100 iter 日志及验证期间的模式和随机状态恢复。
AMP 测试覆盖 FP16/BF16 三模式、动态溢出回退、还原梯度后裁剪、独立分支跳过、
缩放器精确恢复、v3 FP32 迁移及验证的精度隔离；CPU 回退与显式 FP32 逐值一致。
六集验证新增测试覆盖共享官方 LPIPS、原尺寸配对、十八个独立最佳权重、完整指标摘要、
CSV 不重复、验证失败不部分提交、旧单集历史迁移以及自动日志和输出目录校验。
当前 `ild-net` 中 **152 项完整测试全部通过**，包括 CPU/CUDA 及历史源码对照。
重计算开关在 CPU 上逐值比较输出、输入梯度、参数梯度和 AdamW 状态。
CUDA 前向逐值比较，反向使用明确的 FP32 精度容差；反射填充的 CUDA 反向可能
存在非确定性，见 [PyTorch 2.9 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.pad.html)。

本次在 RTX 3090 上使用 8 张真实 SICE 训练图像、512×512、batch 8，
开启激活检查点完成两轮各一步的 CLI 联合训练，每轮完整评价 Part II 250 对。
第一轮后退出，再用 v3 latest 在独立进程恢复第二轮，历史评价和最佳分数完整保留。
第二轮 PSNR/SSIM 提升并更新对应权重，LPIPS 略升因而保留第一轮最佳权重；
四个文件、指标历史、续训进度和计时状态均验证通过。
上述测试用于管线验证，未评估收敛模型的论文质量，临时文件已清理。

六集管线另外在 RTX 3090 上执行一次真实 512×512 AMP 联合训练批次，并完整评价
LOLv1 15 对、LOLv2real 100 对、LOLv2syn 100 对、LSRWhuawei 30 对、
LSRWNikon 20 对、SICE 250 对。18 个最佳权重、6 个实时完整摘要、v5 latest、
6 行指标 CSV 和自动日志均已核验；独立 CLI 恢复到新目录后历史文件逐字节相同，
进度、AMP 缩放及计时不变，未重复验证已完成轮次。此测试使用临时目录和一张
训练图像，模型未收敛；用于确认保存及恢复管线，未启动用户的正式长时间训练。

AMP 验证另外只读加载正在训练的 v3 FP32 检查点（epoch 2、global step 756），
保留进度、优化器及最佳指标历史，在独立进程用一张真实 512×512 SICE 图像执行
两次 AMP 联合更新。双方均完成更新，loss 与参数均有限，scale 保持 64，
测试进程峰值分配约 1169.5 MiB；这是 batch 1 测试，没有在占用显卡的正式训练旁
额外运行 batch 8 大测试。验证未改写用户权重或停止原训练进程。

历史源码的数值对照使用 `legacy_cnn` 和显式 SSIM 范围 255.0；
DEA 及修正后的默认 SSIM 不要求与旧源码逐值相同。原项目归档未修改。
