# DSGCrack evaluation protocols

Two protocols are named and reported separately. All comparisons require the same split, image/mask preprocessing, checkpoint-selection policy and metric definitions.

## Historical paper protocol: legacy

This reproduces the implementation used for the full model and component-ablation tables:

1. Convert each raw logit map to uint8 using round(clip(255 * logits / max(logits), 0, 255)); a zero maximum produces a zero map.
2. Sweep thresholds 0.00, 0.01, ..., 0.99 with strict score/255 > threshold.
3. mIoU: maximum over thresholds of mean per-image foreground/background IoU.
4. ODS: maximum over thresholds of mean per-image F1.
5. OIS: mean of each image's maximum F1.
6. F1: pooled pixel F1 **at historical threshold 0**, after the above logit normalization/quantization.

Historical ODS is an image-averaged score, and historical F1 is not sigmoid F1 at 0.5. Per-image maximum normalization is also not probability calibration. The names are retained to reproduce the reported tables; these values should not be mixed with another evaluator's identically named metrics.

## Probability protocol: sigmoid

Use raw sigmoid(logits), without per-image normalization:

- mIoU: mean per-image two-class IoU at the requested fixed threshold.
- F1 / Precision / Recall: pooled pixel counts at that threshold.
- ODS: maximum pooled/global pixel F1 over thresholds 0.00:0.01:1.00.
- OIS: mean per-image maximum F1 over that grid.

Fixed-threshold metrics operate on raw floating-point probabilities, not saved PNGs. The default mask threshold is 0.5. <code>--calibrate-on-val</code> chooses one threshold per model on validation by pooled pixel F1, breaking ties toward 0.5. Test labels are used to report test ODS/OIS, never to select a checkpoint or saved-mask threshold.

## Preprocessing and selection

Images and masks are resized to 512 × 512. Labels use the original bicubic resize followed by a threshold above 127. Training checkpoints are selected on validation legacy mIoU. Evaluation does not update weights and does not use test-time augmentation.

The supplied filename manifests preserve original sample ordering, including split-specific CamCrack789 numbering. Image stems must be unique within a split. Dataset files and released model weights must be acquired separately.
