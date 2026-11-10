# DSGCrack

**DSGCrack** combines a TransMixer encoder with **Directional Confidence Gating (DCG)** and **Scale-Aware Residual Gating (SRG)** for crack segmentation.

[中文使用说明](docs/README_zh.md) · [Evaluation protocols](docs/evaluation.md) · [Reported results](docs/results.md) · [Source attribution](third_party/README.md)

This release contains the default full model, training, evaluation and image prediction. The model has **2,499,460 parameters**. Input and output resolution is **512 × 512**. Source code uses the names **DCG** and **SRG** consistently.

## Installation

Tested with Linux, Python 3.10, PyTorch 2.7.1, torchvision 0.22.1, CUDA 12.8, and an NVIDIA RTX 5090. A CUDA GPU is required for full-model forward/backward. CPU construction and small component tests do not need the compiled kernel.

Clone the repository, then install from its root:

~~~bash
git clone https://github.com/chenxinxinxin1314-netizen/DSGCrack.git
cd DSGCrack
conda create -n dsgcrack python=3.10 -y
conda activate dsgcrack
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
MAX_JOBS=1 python -m pip install --no-build-isolation ./kernels/selective_scan
~~~

The last command requires a CUDA toolkit containing <code>nvcc</code>; <code>CUDA_HOME</code> must point to that toolkit if it is not discovered automatically. PyTorch chooses the visible GPU architecture by default. For an RTX 5090, <code>TORCH_CUDA_ARCH_LIST="12.0"</code> can be set explicitly with CUDA 12.8. Compiled binaries are built locally and are not distributed.

## Data layout

Bring your own datasets; no images or masks are distributed.

~~~text
data/
  CrackMap/
    train_img/
    train_lab/
    val_img/
    val_lab/
    test_img/
    test_lab/
~~~

Masks are PNG files with foreground values above 127. Image <code>sample.jpg</code> maps to mask <code>sample.png</code>. CamCrack789 uses <code>image-N.png</code> → <code>target-N.png</code>.

The supplied manifests contain basenames in the original experiment ordering. They include 84/12/24 images for CrackMap, 553/79/157 for CamCrack789, 368/53/106 for DeepCrack, and 2357/336/675 for Crack500 (train/validation/test). The program checks manifest consistency. For a different split, provide <code>--split-file your_split.json</code>; <code>--split-file none</code> uses sorted filenames.

Preprocessing reproduces the original full method: RGB, bicubic resizing to 512 × 512, normalization to [-1,1], and train-only paired horizontal/vertical flips and quarter-turn rotations. Masks are resized with the original bicubic-then-threshold procedure. The full-model recipes have no extra photometric augmentation.

## Train

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset CrackMap --data-root data/CrackMap \
  --output outputs/CrackMap
~~~

Replace the dataset name and data path for the other datasets. Recipes use 100 epochs for CamCrack789, CrackMap and DeepCrack, and 50 for Crack500; batch size 1, seed 42, AdamW, learning rate 3e-4, weight decay 0.01, PolyLR to 1e-6, and 0.87 BCE-with-logits + 0.13 Dice loss. Both the encoder's local path and global path are retained.

Only validation is used during training. <code>best.pt</code> is selected by historical validation mIoU; <code>last.pt</code> includes optimizer, scheduler, RNG state and history for resuming:

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset CrackMap --data-root data/CrackMap \
  --output outputs/CrackMap --resume outputs/CrackMap/last.pt
~~~

Epoch count, batch size, seed, threads and loader workers can be overridden through CLI arguments. Algorithm settings and data order must match when resuming. Default precision is FP32; AMP and distributed training are not enabled.

## Evaluate a test set

~~~bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint outputs/CrackMap/best.pt \
  --output outputs/CrackMap/test --protocol both
~~~

All test images are evaluated. <code>metrics.json</code> reports **legacy** and **sigmoid** results in separate fields; values are fractions. Per-image files are saved independently:

- <code>predictions/*_probability.png</code>: uint16 PNG; divide by 65535 for probabilities.
- <code>predictions/*_mask.png</code>: binary sigmoid-threshold mask.
- <code>predictions/*_label.png</code>: resized ground truth.
- <code>legacy/*_pre.png</code>, <code>legacy/*_lab.png</code>: historical scores and labels.

For sigmoid masks with one threshold selected on validation pooled pixel F1:

~~~bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint outputs/CrackMap/best.pt \
  --output outputs/CrackMap/test_calibrated --protocol both \
  --calibrate-on-val
~~~

This selects the threshold on validation before evaluating test images. Test-label threshold sweeps reported as ODS/OIS do not choose the saved mask threshold. See [evaluation.md](docs/evaluation.md) for the historical definitions, including the nonstandard historical F1.

## Predict unlabeled images

~~~bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --input path/to/images --checkpoint weights/model.pt \
  --output outputs/predictions --threshold 0.5
~~~

The input accepts one image or a folder. Each image produces its own probability PNG and mask PNG at 512 × 512.

## Weights

Pretrained checkpoints for the full **DCG + SRG** model are available in [GitHub Release v1.0.0](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/tag/v1.0.0). Model files are distributed as release assets and are not stored in Git history.

| Dataset | Checkpoint | Size |
| --- | --- | ---: |
| CamCrack789 | [DSGCrack_CamCrack789.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CamCrack789.pt) | 9.68 MiB |
| CrackMap | [DSGCrack_CrackMap.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CrackMap.pt) | 9.68 MiB |
| DeepCrack | [DSGCrack_DeepCrack.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_DeepCrack.pt) | 9.68 MiB |
| Crack500 | [DSGCrack_Crack500.pt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_Crack500.pt) | 9.68 MiB |

These are the validation-selected checkpoints for the [original reported results](docs/results.md), converted without changing model tensor values. They contain model tensors and basic configuration only, and load directly without <code>--legacy-checkpoint</code>. Selection epochs and historical test metrics are recorded in [weights_manifest.json](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/weights_manifest.json). Historical metrics use the **legacy** protocol; they should not be interpreted as sigmoid-at-0.5 results.

Download and evaluate, for example:

~~~bash
mkdir -p weights
curl --fail --location \
  https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/DSGCrack_CrackMap.pt \
  --output weights/DSGCrack_CrackMap.pt
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --dataset CrackMap --data-root data/CrackMap \
  --checkpoint weights/DSGCrack_CrackMap.pt \
  --output outputs/CrackMap/pretrained_test --protocol both
~~~

Verify downloaded files against [SHA256SUMS.txt](https://github.com/chenxinxinxin1314-netizen/DSGCrack/releases/download/v1.0.0/SHA256SUMS.txt). Run <code>sha256sum -c SHA256SUMS.txt</code> in the download folder after downloading all four checkpoints.

To prepare a model-only checkpoint for an external release:

~~~bash
python export_checkpoint.py \
  --input outputs/CrackMap/best.pt --output weights/CrackMap.pt
~~~

For trusted original experiment checkpoints:

~~~bash
python export_checkpoint.py --legacy \
  --input path/to/checkpoint_best.pth --output weights/CrackMap.pt
~~~

The adapter strictly maps historical parameter names to the public DCG/SRG names; it does not crop, pad, reshape or discard model tensors. Native evaluation also supports trusted original checkpoints with <code>--legacy-checkpoint</code>. Legacy pickle loading is opt-in; public exports contain only model tensors and basic model metadata, without optimizer/RNG state or machine paths.

## Repository layout

The public Python package is <code>dsgcrack</code>, and the complete model class is <code>DSGCrack</code>:

~~~python
from dsgcrack import DSGCrack
model = DSGCrack()
~~~

~~~text
dsgcrack/
  model.py              # Full DCG + SRG model
  encoder.py            # Four-stage TransMixer encoder
  dcg.py                # Directional Confidence Gating
  srg.py                # Scale-Aware Residual Gating
  decoder.py            # Multilevel projection, SRG, fusion and prediction
  backbone/             # Required TransMixer / VMamba operations
configs/                # Four full-model training recipes
splits/                 # Original filename order; no dataset contents
kernels/selective_scan/ # Required CUDA source
train.py
evaluate.py
predict.py
export_checkpoint.py
tests/
third_party/            # Source notices and license texts
~~~

## Checks

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

The repository does not contain datasets, weights, experiment logs, ablation variants, comparison methods, local environments or compiled binaries. Numerical results from a new training run may vary with hardware and library behavior; fixed split ordering and seed alone do not establish cross-platform bitwise training reproducibility.

## Attribution and license

The TransMixer backbone and supporting operations are adapted from [MixerCSeg](https://github.com/spiderforest/MixerCSeg) and [VMamba](https://github.com/MzeroMiko/VMamba). CUDA selective scan includes work from [Mamba](https://github.com/state-spaces/mamba). Upstream copyright and license notices are retained; see [third_party/README.md](third_party/README.md) and [LICENSE](LICENSE). These acknowledgments identify the inherited backbone separately from the DCG/SRG modifications.
