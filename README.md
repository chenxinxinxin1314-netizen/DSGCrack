# DSGCrack

**DSGCrack** is a crack segmentation framework comprising a TransMixer encoder, **Directional Confidence Gating (DCG)**, and **Scale-Aware Residual Gating (SRG)**. DCG uses local directional statistics to condition residual feature refinement. In SRG, a shared shallow reference is combined with corrections specific to each scale to gate deeper features before fusion. The model contains **2,499,460 parameters** and processes images at **512 × 512** resolution.

[中文说明](docs/README_zh.md) · [Results](docs/results.md) · [Evaluation protocols](docs/evaluation.md) · [Pretrained weights](#weights)

## Installation

The implementation was verified on Linux with Python 3.10, PyTorch 2.7.1, torchvision 0.22.1, CUDA 12.8, and an NVIDIA RTX 5090. Model execution requires an NVIDIA CUDA GPU. The selective scan extension is compiled locally using a CUDA toolkit with `nvcc`.

~~~bash
git clone https://github.com/chenxinxinxin1314-netizen/DSGCrack.git
cd DSGCrack
conda create -n dsgcrack python=3.10 -y
conda activate dsgcrack
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
MAX_JOBS=1 python -m pip install --no-build-isolation ./kernels/selective_scan
~~~

Set `CUDA_HOME` to the CUDA toolkit directory if it is not detected automatically.

## Data preparation

Each dataset is organized as follows:

~~~text
data/CrackMap/
  train_img/
  train_lab/
  val_img/
  val_lab/
  test_img/
  test_lab/
~~~

PNG masks share the image stem, for example `sample.jpg` and `sample.png`. For CamCrack789, `image-N.png` is paired with `target-N.png`. The manifests in `splits/` specify the original sample ordering:

| Dataset | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| CamCrack789 | 553 | 79 | 157 |
| CrackMap | 84 | 12 | 24 |
| DeepCrack | 368 | 53 | 106 |
| Crack500 | 2357 | 336 | 675 |

Input images use uint8 encoding and are converted to RGB, resized by bicubic interpolation, and normalized to [−1, 1]. Masks are resized by bicubic interpolation and binarized at values greater than 127. Training augmentation consists of paired horizontal and vertical flips and rotations in multiples of 90°. Custom splits are supported through `--split-file your_split.json`.

## Weights

The checkpoints in [Release v1.0.0](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/tag/v1.0.0) correspond to the full DCG + SRG model selected by validation mIoU. Each file is approximately **9.68 MiB**.

| Dataset | Download |
| --- | --- |
| CamCrack789 | [DSGCrack_CamCrack789.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CamCrack789.pt) |
| CrackMap | [DSGCrack_CrackMap.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CrackMap.pt) |
| DeepCrack | [DSGCrack_DeepCrack.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_DeepCrack.pt) |
| Crack500 | [DSGCrack_Crack500.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_Crack500.pt) |

Selection epochs and test metrics are provided in [weights_manifest.json](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/weights_manifest.json); file checksums are listed in [SHA256SUMS.txt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/SHA256SUMS.txt).

~~~bash
mkdir -p weights
curl --fail --location \
  https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CrackMap.pt \
  --output weights/DSGCrack_CrackMap.pt
~~~

## Training

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset CrackMap --data-root data/CrackMap \
  --output outputs/CrackMap
~~~

Training configurations are provided in `configs/`: 100 epochs for CamCrack789, CrackMap, and DeepCrack, and 50 epochs for Crack500. AdamW is used with batch size 1, seed 42, learning rate 3 × 10⁻⁴, weight decay 0.01, and polynomial learning rate decay to 10⁻⁶. The loss is `0.87 × BCEWithLogitsLoss + 0.13 × Dice loss`; training uses FP32 precision.

The checkpoint `best.pt` is selected by validation mIoU under the legacy protocol. Training can be resumed by appending `--resume outputs/CrackMap/last.pt` to the command above.

## Evaluation and prediction

Evaluate all test images using the corresponding checkpoint:

~~~bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint weights/DSGCrack_CrackMap.pt \
  --output outputs/CrackMap/test --protocol both
~~~

Paper results follow the **legacy** protocol. The **sigmoid** protocol evaluates probabilities, with a default mask threshold of 0.5. Both are reported separately in `metrics.json` as fractions; their thresholding and aggregation rules are specified in [Evaluation protocols](docs/evaluation.md). Appending `--calibrate-on-val` selects one global threshold by validation pixel F1 for subsequent test predictions.

Probability maps, binary masks, and ground truth masks are saved separately for each test image. Probability PNGs are encoded as uint16; divide pixel values by 65535 to convert them to probabilities in [0, 1].

For an image or a folder of images:

~~~bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input path/to/images --checkpoint weights/DSGCrack_CrackMap.pt \
  --output outputs/predictions --threshold 0.5
~~~

## Implementation and license

The model is implemented in `dsgcrack/`, with DCG in `dcg.py`, SRG in `srg.py`, and feature fusion in `decoder.py`. `export_checkpoint.py` exports model tensors and configuration for distribution.

The TransMixer backbone and supporting operations are adapted from [MixerCSeg](https://github.com/spiderforest/MixerCSeg) and [VMamba](https://github.com/MzeroMiko/VMamba). The CUDA selective scan implementation incorporates code from [Mamba](https://github.com/state-spaces/mamba). Copyright and license notices are provided in [third_party/README.md](third_party/README.md) and [LICENSE](LICENSE).
