# Third-party notices

Project-owned code is MIT licensed. This license does not relicense NVIDIA or Microsoft runtimes or model components. Retain the component notices in this package.

- DirectShow transport and scheduler derive from [nvof-potplayer](https://github.com/mkyoon80-alt/nvof-potplayer), commit `12b5c2622a717cbe35625ecfb2f1b5ad4fcbfba2`. Its MIT notice remains in LICENSE.
- Microsoft DirectShow baseclasses: `licenses/Microsoft-Windows-Samples-MIT.txt`. The vendored source keeps its original notices.
- Self-contained .NET 10 / WPF and application-local Microsoft runtime libraries: corresponding Microsoft licenses and third-party notices in `licenses/`.
- TensorRT 10.16.0.72 runtime DLLs and builder resource DLLs: [NVIDIA TensorRT SDK agreement](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/reference/sla.html), included as `licenses/NVIDIA-TensorRT-10-SDK-agreement.html`. No NVIDIA display driver or SDK developer executable is bundled. The offline `RifeEngineBuilder.exe` is built from this project's source.
- TensorRT headers and precision-rule reference: NVIDIA TensorRT `release/10.16`, under `licenses/TensorRT-Source-LICENSE.txt`. This software contains source code provided by NVIDIA Corporation. Source terms do not replace the NVIDIA binary SDK terms.
- CUDA 13-family CUDART/NVRTC runtime components: `licenses/NVIDIA-CUDA-13-EULA.html`. These proprietary components may only be used and redistributed as permitted by the vendor terms, as application components; they are not relicensed under MIT.
- RIFE 4.25 Standard: [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) / [RIFE](https://github.com/hzwer/ECCV2022-RIFE), copyright (c) 2021 hzwer, MIT notice in `licenses/RIFE-Source-MIT.txt`. The ONNX export was verified against `rife_v2/rife_v4.25.onnx` in the [vs-mlrt external-models release](https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models). Original SHA256: `65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584`; motion/mask export SHA256: `95c59f2b45ab358981e7ec044b946abd69960e29d034bc16e99223768b284506`. The extractor keeps internal warps and changes only the exported output/pruned RGB synthesis. No VapourSynth/vs-mlrt plugin or Smootter executable is included.
- Three manual screenshots were supplied by the project user for the RIFE release documentation. See `docs/screenshot-provenance.json`.

This project is independent of NVIDIA, Microsoft, PotPlayer and Smootter. No private videos, playback captures, credentials or local diagnostic logs are distributed.
