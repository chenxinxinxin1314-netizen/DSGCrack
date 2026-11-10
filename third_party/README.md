# Source attribution and retained licenses

The release preserves the original default forward computation, while removing unused model families and adopting DCG/SRG module names.

| Component | Immediate source | Retained notice |
|---|---|---|
| TransMixer and encoder/decoder building blocks | [MixerCSeg](https://github.com/spiderforest/MixerCSeg) | [MixerCSeg MIT](licenses/MixerCSeg-MIT.txt) |
| Scan utilities and TransMixer's VMamba ancestry | [VMamba](https://github.com/MzeroMiko/VMamba) | [VMamba MIT](licenses/VMamba-MIT.txt) |
| CUDA selective-scan code and copyright headers | VMamba's oflex kernel, derived from [Mamba v1.2.0](https://github.com/state-spaces/mamba/tree/v1.2.0) | [Mamba Apache-2.0](licenses/Mamba-Apache-2.0.txt) |
| CUB-derived helper code in CUDA headers | [NVIDIA CUB](https://github.com/NVIDIA/cub/tree/1.16.0) | [CUB BSD-3-Clause](licenses/CUB-BSD-3-Clause.txt) |

Upstream notices remain in the CUDA sources. The original MixerCSeg repository also acknowledges SCSegamba, HiddenMambaAttn, LongMamba and DeciMamba; inherited contributions are not claimed as new DCG/SRG implementations.

Changes include semantic DCG/SRG naming, a strict name-only legacy checkpoint adapter, removal of unused model variants, a source-only CUDA build entry point, portable training/evaluation CLIs, and explicit metric protocols. No pretrained values are altered by the name conversion.

The root MIT license is retained from the immediate source; bundled third-party components retain their own applicable notices and license texts.

Original backbone citation:

~~~bibtex
@article{zhao2026mixercseg,
  title={MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention},
  author={Zhao, Zilong and Ding, Zhengming and Niu, Pei and Sun, Wenhao and Guo, Feng},
  journal={arXiv preprint arXiv:2603.01361},
  year={2026}
}
~~~

The full DSGCrack paper title, author list and final citation can be added when publication details are available.
