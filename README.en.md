# RIFE for PotPlayer

Real-time 8K 240fps interpolation for PotPlayer, powered by RIFE and native D3D11. Tested at 8K 60fps ×4 on RTX 5090.

**v0.2.1 · Windows x64 · RIFE 4.25 Standard**

**GPU support: NVIDIA GeForce RTX 50 series only.** RTX 5090 is tested; normal operation on RTX 5080 has been confirmed by the user. RTX 30 series did not work in the user’s test and is unsupported. Other GPU series are outside this release’s support scope. The 8K 60fps ×4 performance result applies to RTX 5090 only.

[Download the standalone ZIP](https://github.com/mkyoon80-alt/rife-potplayer/releases/tag/v0.2.1) · [한국어](README.md) · [English HTML manual](https://mkyoon80-alt.github.io/rife-potplayer/manual.en.html) · [한글 HTML 매뉴얼](https://mkyoon80-alt.github.io/rife-potplayer/)

The release ZIP includes the .NET/WPF runtime, NVIDIA inference runtime, model, engine preparation tool and both offline HTML manuals. Extract the entire folder and open **RifeControl.exe**. No Python, VapourSynth or separate .NET installation is required. PotPlayer x64 and a compatible NVIDIA driver are required. The application UI is currently Korean; the English manual explains every main control.

## Connect

1. Open **연결 설정 → 엔진 검사 / 준비** (Connection → Check / prepare engine).
2. Register the filter from the control app. Elevated PotPlayer requires administrator registration.
3. In PotPlayer's **Preferences → Filter Control → Global Filter Priority**, add **RIFE for PotPlayer**, enable it and select highest priority / Prefer. Do not enable another interpolation filter at the same time.
4. Reopen a video and check the active state and input → output FPS. The default output cap is 60fps.

Registration alone does not select the filter in PotPlayer. Read the online manuals using the links above, or open `docs/manual.en.html` or `docs/manual.ko.html` locally after extracting for offline reading.

## Processing

RIFE estimates motion and blending masks on a fixed guide with a maximum long edge of **960**, using **TensorRT FP16** with FP32 coordinates and selected operations. The filter composes the **original-resolution pixels** on D3D11. It does not enlarge a completed 960-pixel RGB image. NV12/P010, SDR/HDR color information, software input conversion and hardware/software reconnect paths are covered by automated checks.

An engine is prepared for the computer and reused across video files. The playback filter never compiles TensorRT engines on file open or seek. The package includes the previously validated RTX 5090 plan; an incompatible GPU/runtime needs preparation through the separate tool. This release includes SM120/PTX builder resources for its RTX 50-series support scope. Engine preparation is for supported GPUs; it does not extend support to RTX 30 or other series.

## Performance scope

The 8K 60→240fps result is a real PotPlayer test on an RTX 5090, not a guarantee for every source or configuration. Startup drops remain. VSR (GPU Super Resolution) and high interpolation rates can contend for GPU time and cause drops during motion. Both are tracked as longer-term work. A 165Hz display cannot present all 240 distinct output frames per second.

Startup preparation stays at **100ms of wall-clock time**, showing originals while audio continues. The seek original-preview threshold is **50ms**; actual resumption also depends on frame cadence and a new adjacent pair. Detailed diagnostics remain enabled, sampled every 32 input pairs. UI live updates now preserve the reading position and text selection.

HDR10/HLG metadata paths are tested. Dolby Vision coverage is decoder-provided metadata forwarding, not full profile/FEL/display certification. HDR10+ dynamic metadata forwarding is not implemented. See the manuals for color-format limitations.

## Development and credits

v0.2.1 adds a **160fps** cap to both common and per-resolution output settings and fixes source FPS being doubled for some progressive HEVC 10-bit videos. A 23.976fps source previously reported as 47.952fps now correctly interpolates **23.976 → 119.88fps** at a 120fps cap. Actual interlaced inputs retain their existing processing path.

The 160fps option is an integer-multiple cap: 24fps outputs 144fps, 30fps outputs 150fps, and 60fps outputs 120fps. It does not change the monitor refresh rate. The FPS issue was reproduced with PotPlayer's **F5 → Video → ‘10-bit output’ beside Video buffer format** enabled. This update does not force 10-bit output or HDR; keep your existing settings and reopen the video.

Before updating, close PotPlayer and RifeControl and back up the old folder. Extract the new ZIP separately and copy the old `RifeStandardPotPlayer.ini` into it to retain settings. Unregister from the old folder, then check the engine and register from the new folder. See the manual for the complete update steps.

[Build instructions](docs/BUILDING.md) · [Changes](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

The DirectShow transport and scheduler derive from [nvof-potplayer](https://github.com/mkyoon80-alt/nvof-potplayer), starting at commit `12b5c2622a717cbe35625ecfb2f1b5ad4fcbfba2`. RIFE uses its own filter IDs and configuration. The original NVOF repository and installation are independent. Some internal `nvof` namespaces remain for transport code compatibility; the RIFE filter does not run the NVOF interpolation engine.

Project code is MIT licensed. Third-party runtimes and models retain their own terms. This project is independent of NVIDIA, Microsoft, PotPlayer and Smootter.
