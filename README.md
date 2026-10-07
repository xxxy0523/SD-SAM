# SD-SAM

**Dataset CPD1K:** [https://www.kaggle.com/datasets/haoranyang20030523/cpd1k-a-colon-polyp-segmentation-dataset](https://www.kaggle.com/datasets/haoranyang20030523/cpd1k-a-colon-polyp-segmentation-dataset)

SD-SAM 是一个两阶段息肉图像分割训练项目。第一阶段使用冻结的 DINOv3 特征训练 Gaussian prompt decoder，自动生成点提示；第二阶段加载第一阶段的最佳提示器权重，并在冻结基础模型的同时训练 SAM 适配器、深层 QKV/特征融合模块和蒸馏投影。

当前默认实验使用 Kvasir-SEG 和 CVC-ClinicDB 训练，在二者的留出部分以及 CVC-ColonDB、CVC-300 和 ETIS 上进行评估。CPD1K 的公开下载地址见本页开头，当前默认训练配置中未启用 CPD1K。

## 项目结构

```text
SD-SAM/
├── train_stage1.py          # 第一阶段训练入口，配置集中在文件顶部 CONFIG
├── train_stage2.py          # 第二阶段训练入口，配置集中在文件顶部 CONFIG
├── selfprompt/              # Gaussian prompt decoder
├── model/                   # 数据集、损失函数和完整 SD-SAM 模型
├── segment_anything/        # 项目使用的 SAM 实现
├── dinov3/                  # 项目使用的 DINOv3 实现
├── utils/                   # 数据划分、权重保存和实验记录工具
├── licenses/                # 第三方许可证
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

所有实验配置都位于 `train_stage1.py` 和 `train_stage2.py` 顶部的 `CONFIG` 中。训练生成的 `config.json` 只用于保存当次实验的实际配置，不是额外的配置输入文件。

## 环境安装

建议使用 Python 3.10 或更高版本。根据本机 CUDA 环境安装匹配的 PyTorch 和 torchvision，然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
```

本项目整理时使用的版本为 PyTorch 2.8.0 和 torchvision 0.23.0。仓库已经包含训练所需的 SAM 与 DINOv3 源码，不需要另外安装 `segment-anything`、`dinov3`、`timm` 或 `scikit-learn`。

## 第一阶段：训练自动点提示器

第一阶段冻结 DINOv3，只训练 Gaussian prompt decoder。默认训练 50 轮，batch size 为 2，优化目标为：

```text
10 × MSE + BCE
```

先检查数据路径、DINOv3 权重和数据划分：

```bash
python train_stage1.py --check
```

开始训练：

```bash
python train_stage1.py
```

代码默认每 5 轮评估一次；如果最后一轮不是 5 的倍数，最后一轮仍会进行评估。第一阶段使用预测点是否落在目标掩码内计算 `hit_rate`，并以五个评估数据集的平均 Hit Rate 选择最佳权重：

```text
outputs/stage1/prompt_decoder.pth
```

## 第二阶段：训练分割模型

第二阶段自动加载以下三个权重：

```text
checkpoints/sam_vit_b_01ec64.pth
checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
outputs/stage1/prompt_decoder.pth
```

第一阶段提示器权重会被冻结。第二阶段默认训练 50 轮，batch size 为 1，并使用分割损失和 DINOv3 特征蒸馏损失训练可学习模块。

配置项 `W=5` 表示编号满足 `i < W` 的前 5 个 SAM block 使用浅层适配与蒸馏，满足 `i >= W` 的其余 block 使用深层融合。QKV 融合按照当前代码使用普通线性层。

先检查所有路径、共享划分以及第一阶段权重中的元数据：

```bash
python train_stage2.py --check
```

开始训练：

```bash
python train_stage2.py
```

第二阶段同样默认每 5 轮评估一次，分别计算每个数据集的 Dice 和 IoU，并以五个评估数据集的平均 Dice 选择最佳权重：

```text
outputs/stage2/model.pth
```

## 指标与训练历史

训练过程中的指标保存在以下文件中：

```text
outputs/stage1/history.json
outputs/stage2/history.json
```

`history.json` 在每一轮训练结束后更新。普通训练轮次只记录训练损失；第 5、10、15……轮以及最后一轮还会在 `evaluation` 字段中记录每个留出评估集和泛化评估集的指标。

第一阶段记录格式示例：

```json
{
  "epoch": 5,
  "loss": 0.1234,
  "evaluation": {
    "Kvasir": {"hit_rate": 0.92, "hits": 92, "samples": 100},
    "ClinicDB": {"hit_rate": 0.90, "hits": 90, "samples": 100},
    "ColonDB": {"hit_rate": 0.86, "hits": 326, "samples": 380},
    "CVC-300": {"hit_rate": 0.88, "hits": 53, "samples": 60},
    "ETIS": {"hit_rate": 0.82, "hits": 161, "samples": 196}
  },
  "average_hit_rate": 0.876
}
```

第二阶段记录格式示例：

```json
{
  "epoch": 5,
  "loss": 0.2345,
  "segmentation": 0.2100,
  "distillation": 0.4900,
  "evaluation": {
    "Kvasir": {"dice": 0.91, "iou": 0.84, "samples": 100},
    "ClinicDB": {"dice": 0.89, "iou": 0.81, "samples": 100},
    "ColonDB": {"dice": 0.78, "iou": 0.68, "samples": 380},
    "CVC-300": {"dice": 0.82, "iou": 0.72, "samples": 60},
    "ETIS": {"dice": 0.76, "iou": 0.65, "samples": 196}
  },
  "average_dice": 0.832,
  "average_iou": 0.740
}
```

示例数字仅用于说明 JSON 结构，不代表实际实验结果。`average_hit_rate`、`average_dice` 和 `average_iou` 都是五个数据集指标的等权平均值，不按各数据集样本数量加权。

训练完成后还会生成：

```text
outputs/stage1/test_metrics.json
outputs/stage2/test_metrics.json
```

- 第一阶段的 `test_metrics.json` 保存平均 Hit Rate 最佳轮次对应的各数据集指标和 `_selection` 信息。
- 第二阶段的 `test_metrics.json` 保存平均 Dice 最佳轮次对应的各数据集指标、`_selection` 信息，以及 `_best_per_dataset`。其中 `_best_per_dataset` 记录每个数据集在整个评估过程中达到的最高 Dice、该轮对应的 IoU 和 epoch。
- 每个数据集的历史最高指标只作为数值记录；程序不会为每个数据集分别保存一份最佳模型权重。

当前代码按照五个评估数据集的等权平均指标选择 checkpoint，因此 CVC-ColonDB、CVC-300 和 ETIS 也参与最佳模型选择。这与本仓库保留的原始训练逻辑一致。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `outputs/splits/five_datasets.json` | 两阶段共享的数据划分、文件清单和摘要 |
| `outputs/stage1/config.json` | 第一阶段实际运行配置 |
| `outputs/stage1/parameters.json` | 第一阶段参数量统计 |
| `outputs/stage1/history.json` | 第一阶段逐轮损失和每次评估结果 |
| `outputs/stage1/prompt_decoder.pth` | 平均 Hit Rate 最佳的第一阶段提示器权重 |
| `outputs/stage1/last.pth` | 第一阶段最后一轮权重和优化器状态 |
| `outputs/stage1/test_metrics.json` | 第一阶段最佳平均指标对应的汇总结果 |
| `outputs/stage2/config.json` | 第二阶段实际运行配置 |
| `outputs/stage2/parameters.json` | 第二阶段参数量统计 |
| `outputs/stage2/history.json` | 第二阶段逐轮损失和每次评估结果 |
| `outputs/stage2/model.pth` | 平均 Dice 最佳的第二阶段模型权重 |
| `outputs/stage2/last.pth` | 第二阶段最后一轮权重和优化器状态 |
| `outputs/stage2/test_metrics.json` | 最佳平均指标及各数据集历史最高指标汇总 |

`data/`、`checkpoints/`、`outputs/`、模型权重、IDE 文件和 Python 缓存均已加入 `.gitignore`，不会随代码上传到 GitHub。需要保留实验结果时，请在本地备份 `outputs/`。

## 第三方代码与许可证

SAM 和 DINOv3 的来源及许可证信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses](licenses/) 目录。
