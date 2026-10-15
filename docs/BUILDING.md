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

The v0.2.0 filter includes the validated playback cleanup: asynchronous playback status publication, reused source plane views, and removal of the unused generic GPU timeline path. Pixel synthesis, scheduling, 100ms startup preparation and the 50ms seek-preview threshold are preserved. Native/UI file version is 0.2.0.0; release version is 0.2.0. The distributed runtime supports RTX 50 series only: RTX 5090 is tested and RTX 5080 normal operation is user-confirmed. RTX 30 series is unsupported.
