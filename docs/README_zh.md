# DSGCrack 中文说明

**DSGCrack** 由 TransMixer 编码器、方向置信门控（DCG）和尺度感知残差门控（SRG）组成，用于裂缝分割。DCG 根据局部方向统计量调节残差特征；SRG 将共享浅层参考与各尺度修正项结合，对深层特征进行门控后融合。模型包含 **2,499,460** 个参数，输入与输出分辨率均为 **512 × 512**。

## 安装与数据

安装步骤和数据划分见[根目录 README](../README.md)。已验证的环境为 Linux、Python 3.10、PyTorch 2.7.1、torchvision 0.22.1、CUDA 12.8 和 NVIDIA RTX 5090。模型运行需要 NVIDIA CUDA GPU，选择性扫描算子使用本机 `nvcc` 编译。

每个数据集应包含 `train_img`、`train_lab`、`val_img`、`val_lab`、`test_img` 和 `test_lab` 六个目录。PNG 掩膜通常与图像同名；CamCrack789 中 `image-N.png` 对应 `target-N.png`。`splits/` 中的清单指定原实验的样本顺序。

## 权重与训练

四个数据集的完整模型权重已发布至 [Release v1.0.0](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/tag/v1.0.0)，每份约 **9.68 MiB**，下载入口见[权重表](../README.md#weights)。附件提供所选 epoch、测试指标和 SHA256 校验值。

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset CrackMap --data-root data/CrackMap \
  --output outputs/CrackMap
~~~

训练配置位于 `configs/`。CamCrack789、CrackMap 和 DeepCrack 训练 100 个 epoch，Crack500 训练 50 个 epoch。最佳权重 `best.pt` 根据验证集的 legacy mIoU 选取；续训时添加 `--resume outputs/CrackMap/last.pt`。

## 测试与预测

以下命令使用下载至 `weights/DSGCrack_CrackMap.pt` 的预训练权重：

~~~bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint weights/DSGCrack_CrackMap.pt \
  --output outputs/CrackMap/test --protocol both
~~~

论文结果采用 **legacy** 评估协议；**sigmoid** 协议基于概率计算指标，默认掩膜阈值为 0.5。两套指标分别写入 `metrics.json`，阈值处理和聚合定义见[评估说明](evaluation.md)。添加 `--calibrate-on-val` 后，将依据验证集像素 F1 选择一个全局阈值，并用于测试集预测。

每张测试图像分别保存概率图、二值掩膜和真值掩膜。概率图采用 16 位 PNG 编码，像素值除以 65535 即为概率。

单张图像或图像文件夹的预测命令为：

~~~bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input path/to/images --checkpoint weights/DSGCrack_CrackMap.pt \
  --output outputs/predictions --threshold 0.5
~~~

## 来源与许可

上游来源与许可见 [third_party/README.md](../third_party/README.md) 和 [LICENSE](../LICENSE)。
