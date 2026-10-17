# Build and validation

Use Windows x64 with Visual Studio 2022 C++ tools, Windows SDK and CUDA headers. CMake 3.20+ is required. `ui/dotnet-sdk.json` pins the SDK/runtime used for the self-contained WPF app; `ui/get-dotnet-sdk.ps1` can acquire the SDK for development. Playback does not need these development tools.

The repository vendors the Microsoft DirectShow baseclasses and TensorRT 10.16 headers used by the build, with their licenses. Inference runtime DLLs, models, prepared engines, video captures and local logs are release assets or local inputs, not Git source files.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass -File ui/build-ui.ps1 -OutputDirectory bin
bin/RifeControl.exe --self-test --runtime-info build/ui-runtime.json
bin/RifeControl.exe --test-diagnostics-scroll build/ui-scroll.json
```

CUDA headers can be selected with `-DCUDA_HEADERS=<CUDA include directory>`. GPU tests require a compatible NVIDIA GPU and the release payload placed under `bin/runtime`, `bin/models`, and `bin/engines`. Local scene-fixture tests are registered only when their captured grid data exists; private video fixtures are not shipped in Git.

`Build-Product.ps1` builds the native targets and UI and assembles the local `bin` folder. Keep an existing runtime/model payload available. `tools/rife_engine_builder.cpp` builds the separate `RifeEngineBuilder.exe`; it is not loaded by the playback filter. `Prepare-Engine.ps1` checks and reuses a compatible plan, or prepares a candidate offline and validates the D3D11 inference path with rollback on failure.

The precision policy matches the prior TensorRT CLI build: fixed `input:1x7x540x960`, FP16 enabled, TF32 disabled, obey precision constraints, selected coordinate/mask layers FP32, FP32 linear I/O, optimization level 3. Shape/index layers are excluded from the wildcard FP32 constraint, matching TensorRT 10.16 sample behavior.

The original RIFE export is `rife_v2/rife_v4.25.onnx` from the [vs-mlrt external models release](https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models), SHA256 `65c57a5e4abb17ad67faf35054291ac53affab0506d509d2e66698f6ecd75584`. `tools/extract_rife_motion.py` retains internal iterative warps and exports the final motion/mask tensors, pruning only final low-resolution RGB synthesis. Result SHA256: `95c59f2b45ab358981e7ec044b946abd69960e29d034bc16e99223768b284506`. Extracting the model is a developer operation requiring Python/ONNX; playback and ordinary engine preparation do not use Python.

The v0.3.0 playback filter enables the validated E03/E04 build option by default. E03 reuses features within multi-output source pairs; E04 is gated by a source long edge of at least 3840. The full engine starts interpolation while optional plans load. Required warmup completion, the original-video/audio lead-in and unchanged output timestamps are separate concerns. Native/UI version is 0.3.0.0. The 160fps cap, HEVC correction, actual interlace processing and 50ms seek preview are retained; progressive AV1 P010 negotiation is also corrected.

`RifeSplitEngineBuilder.exe` prepares the fixed-shape Head/Tail models with the same FP16 / critical-FP32 policy. `Prepare-Engine.ps1` checks all three plans and prepares missing/incompatible optional plans outside playback. Both optional candidates must build before publication. Failed validation rolls back their previous files. `RifeEngineCheck --check-optimized bin/runtime` requires actual E03 execution, checks the E04 size policy, and compares six NV12/P010 outputs exactly against the full engine. If optional plans are unavailable, playback retains full-engine interpolation.

Split model SHA256: Head `1fd11285c820848c2deac5ce3758bf7d45a54cb1433e86e3d786c9d707245ec4`; Tail `0af2cbccff025d6f5c10169f7f50026651580d58c2a46a622329023d5c693701`. `tools/extract_rife_split.py` reproduces the boundary from the unchanged motion model; Python/ONNX are developer dependencies only. Prepared engines are GPU/runtime dependent. Support remains RTX 50 series; the new optimization performance measurements are from RTX 5090, not a new RTX 5080 performance claim.
