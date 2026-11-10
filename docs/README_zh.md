# DSGCrack 中文使用说明

本目录发布 **DSGCrack** 的默认完整方法：**TransMixer 编码器 + DCG + SRG**。主模型类为 DSGCrack，Python 包为 dsgcrack，两个模块分别命名为 DCG、SRG；不包含消融、超参数实验、MoE、对比模型、数据集、权重和训练日志。

## 安装

按根目录 README 安装 Python 3.10、PyTorch 2.7.1 / torchvision 0.22.1（CUDA 12.8），再安装依赖和 CUDA 算子。完整模型需要 NVIDIA CUDA GPU；算子以源码形式提供，需要本机 nvcc 编译。

## 数据与训练

每个数据集使用 train_img/train_lab/val_img/val_lab/test_img/test_lab 六个目录。splits 文件夹只包含原实验图片文件名顺序，不含图片。

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset CrackMap --data-root data/CrackMap --output outputs/CrackMap
~~~

CamCrack789、CrackMap、DeepCrack 默认训练100 epoch；Crack500为50 epoch。默认 batch1、seed42、lr3e-4、几何增强、无额外光度增强。只使用验证集选最佳权重。

## 测试与逐张预测

~~~bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint outputs/CrackMap/best.pt \
  --output outputs/CrackMap/test --protocol both --calibrate-on-val
~~~

测试集每张图分别输出概率图、mask和GT。旧论文口径保存在 legacy 指标字段及同名子目录；sigmoid口径保存在 sigmoid 字段。两套指标分开记录。

校准开关在验证集选择一个全局阈值，随后用于测试出图；不根据测试集挑选阈值。概率图为16位PNG，除以65535还原到[0,1]。

~~~bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input path/to/images --checkpoint weights/CrackMap.pt \
  --output outputs/predictions --threshold 0.5
~~~

预测不需要GT，单张图或文件夹均可，输出大小为512×512。

## 旧权重与后续发布

仓库不放权重，后续在根目录 README 添加真实下载链接。旧实验权重可用 export_checkpoint.py 的 legacy 开关转成仅含模型张量和基本配置的公开权重；转换严格校验键名和形状，不裁剪或补齐权重。

公开版保留原版权和第三方许可。核心文件为 dsgcrack/dcg.py、dsgcrack/srg.py、dsgcrack/encoder.py 和 dsgcrack/decoder.py；旧名称仅用于权重兼容映射与上游来源说明。

历史评估的 F1 与 sigmoid@0.5 的 F1 定义不同，详见 [evaluation.md](evaluation.md)。复现实验中必须说明所用评估口径。
