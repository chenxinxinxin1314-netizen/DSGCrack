# DSGCrack reported results

Historical test results for DSGCrack, in percent. These are reference records, not new training runs produced during source packaging. Weight files are not bundled.

| Dataset | Epochs | Test images | mIoU | ODS | OIS | F1 |
|---|---:|---:|---:|---:|---:|---:|
| CamCrack789 | 100 | 157 | 85.44 | 83.04 | 84.32 | 84.62 |
| CrackMap | 100 | 24 | 82.17 | 79.15 | 80.36 | 77.86 |
| DeepCrack | 100 | 106 | 91.80 | 91.28 | 92.17 | 92.42 |
| Crack500 | 50 | 675 | 78.82 | 73.73 | 75.10 | 78.10 |

The default has 2,499,460 parameters. The encoder channels are [16,32,64,128], with one TransMixer + DCG stage block at each level. DCG uses 2×2 spatial partitions and 8×8 statistical cells throughout. SRG uses an 8-channel shared projection and feature-conditioned residual gate corrections.

The table follows the historical legacy protocol defined in [evaluation.md](evaluation.md). In particular, historical F1 is not sigmoid F1 at 0.5, and historical ODS averages per-image F1 before threshold maximization. Sigmoid results are available from the separate evaluation protocol and must not be inserted into this table under the historical labels.

All checkpoints were selected by validation mIoU, with no test-time augmentation. Split manifests and full-model recipes are included. A new training run may differ numerically with hardware/library behavior; reproduction should report the actual checkpoint-selection and metric protocol used.
