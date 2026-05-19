# 第三次作业：LeNet-5 预训练与垃圾分类微调

这个工程用于完成“手写数字识别与垃圾分类”作业。主线是：

1. 在 MNIST 手写数字数据集上训练 LeNet-5，保存预训练权重。
2. 加载 MNIST 预训练权重，替换最后分类层。
3. 在垃圾分类数据集上做冻结微调和全量微调。
4. 输出准确率、训练时间和 checkpoint，作为实验报告素材。

## 1. 创建 conda 环境

```bash
cd /Users/fish37/Desktop/人工智能课程/第三次作业
conda env create -f environment.yml
conda activate ai_hw3_ms
```

如果环境已经创建过，只需要：

```bash
conda activate ai_hw3_ms
```

## 2. 训练 MNIST 预训练模型

先小样本试跑，确认环境正常：

```bash
python -m src.train_mnist \
  --epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 128 \
  --batch-size 32
```

正式训练：

```bash
python -m src.train_mnist --epochs 2 --batch-size 64
```

输出文件：

- `checkpoints/lenet5_mnist.ckpt`
- `outputs/mnist_metrics.json`

## 3. 下载并准备垃圾分类数据集

脚本可以自动下载老师给的数据集：

```bash
python -m src.finetune_garbage --download --epochs 5 --mode both
```

如果你已经手动下载了 `data_en.zip`，把它放到：

```text
data/data_en.zip
```

然后运行：

```bash
python -m src.finetune_garbage --epochs 5 --mode both
```

输出文件：

- `checkpoints/lenet5_garbage_frozen.ckpt`
- `checkpoints/lenet5_garbage_full.ckpt`
- `outputs/garbage_metrics.json`

## 4. 生成报告用结果摘要

```bash
python -m src.summarize_results
```

结果会写入：

```text
outputs/result_summary.md
```

## 5. 运行改进模型

在完成 LeNet-5 迁移学习 baseline 后，可以运行改进模型实验：

```bash
python -m src.train_improved_garbage --epochs 5 --batch-size 32 --lr 0.0005
python -m src.summarize_results
```

改进模型使用 RGB 64×64 输入，增加卷积通道、BatchNorm 和 Dropout。训练轮数、batch size、学习率与 LeNet-5 垃圾分类实验保持一致，便于报告中做前后对比。

## 6. 报告里可以这样解释

本实验先在 MNIST 数据集上训练 LeNet-5，得到卷积层和全连接层的初始特征提取能力。随后将模型最后一层从 10 类手写数字分类替换为垃圾分类类别数，并在垃圾分类数据集上进行迁移学习。实验比较了两种微调方式：

- 冻结微调：冻结 `conv1`、`conv2`、`fc1`、`fc2`，只训练新的 `fc3` 分类层。
- 全量微调：加载预训练权重后，所有层都参与训练。

因为 MNIST 是灰度 32×32 输入，而垃圾图片通常是彩色大图，本工程将垃圾图片统一转成灰度并 resize 到 32×32，使输入维度与 LeNet-5 保持一致。这个做法牺牲了一部分颜色信息，但能清楚体现“加载预训练模型并迁移到新任务”的过程。

模型改进部分使用 RGB 图片和更深的 CNN，以保留颜色和局部纹理信息。报告中应说明改进后准确率提升，同时训练时间和内存占用也增加。
