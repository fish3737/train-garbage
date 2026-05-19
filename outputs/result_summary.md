# 实验结果摘要

## MNIST 预训练
- 训练轮数：2
- 验证准确率：0.9831
- 训练耗时：28.28 秒
- 保存权重：`checkpoints/lenet5_mnist.ckpt`

## 垃圾分类微调
- 类别数：26
- frozen：验证准确率 0.0903，训练耗时 3.18 秒，峰值内存 398.81 MB，权重 `checkpoints/lenet5_garbage_frozen.ckpt`
- full：验证准确率 0.1912，训练耗时 3.44 秒，峰值内存 408.17 MB，权重 `checkpoints/lenet5_garbage_full.ckpt`

## 改进模型
- 模型：ImprovedGarbageCNN
- 输入：RGB 64x64
- 验证准确率：0.4726，训练耗时 65.87 秒，峰值内存 720.89 MB
- 保存权重：`checkpoints/improved_garbage_cnn.ckpt`
