# Changelog

## v0.2.1 — 2026-10-17

- Add the 160fps cap to common and per-resolution output settings. Integer-multiple behavior is preserved: 24→144, 30→150, 60→120fps; this setting does not change monitor refresh rate.
- Fix doubled source FPS for progressive HEVC/P010 video incorrectly advertised as interlaced by the upstream decoder, including the reproduced PotPlayer built-in FFmpeg / 10-bit output case. At a 120fps cap, the affected 23.976fps source now outputs 119.88fps instead of 95.904fps.
- Apply the correction only when HEVC sequence headers explicitly establish progressive-only content. Actual interlaced, mixed, unspecified, contradictory and malformed headers retain the existing behavior. A genuine 47.952fps source is not blindly halved.
- Preserve RIFE 4.25 Standard, TensorRT FP16, 960 guide, original-resolution composition, 10-bit/HDR paths, 100ms startup preparation and 50ms seek preview. No settings migration or default changes are required.
- Regression comparison of the fix passed 29 local automated tests and 59 DirectShow graph cases, including actual decoded HEVC/H.264 interlaced samples; 800 interpolation hashes, 1,196 sequence output hashes and 12 scene-cut source checks matched their references. This is filter-level coverage, not a new end-to-end PotPlayer performance claim. Correcting the affected clip increases intended output work from 95.904 to 119.88fps (+25%).
- Update Korean/English manuals with the 160fps option, HEVC FPS troubleshooting, the exact PotPlayer 10-bit checkbox and settings-preserving update instructions. Native/UI version is 0.2.1.0.

## v0.2.0 — 2026-10-15

- First stable release. GPU support is limited to RTX 50 series: RTX 5090 tested; RTX 5080 normal operation user-confirmed. RTX 30 series is unsupported.
- Updated Korean/English manuals and separate offline HTML assets, including the user-provided RTX 5090 8K playback capture (VSR OFF; approximately 265W at capture).

- Keep playback state publication asynchronous; only final Stop waits for the latest status snapshot to reach disk.
- Reuse owned NV12/P010 source plane views during RIFE preparation, with the existing fallback for external surfaces.
- Remove the unused generic GPU timeline/capture path; retain sampled RIFE TensorRT, CUDA, D3D11 and CPU diagnostics.
- Preserve startup preparation (100ms), seek preview (50ms), output scheduling, renderer sample ownership and interpolation math.
- Validated with 28 automated tests, 800 identical output hashes from 20 source-video pairs and 3 additional cut cases, and DirectShow startup/seek/lifecycle/HDR checks. Renderer/VSR drop counts require a separate PotPlayer comparison.

## v0.1.0-beta — 2026-10-15

- First public standalone Windows x64 release of RIFE for PotPlayer.
- RIFE 4.25 Standard, TensorRT FP16, fixed 960 guide and original-resolution D3D11 composition.
- Per-computer engine preparation; reuse across source files without playback-time engine compilation.
- Hardware/software input and reconnect handling, HDR metadata paths, scene-cut handling and seek cancellation.
- Retains the validated playback filter, 100ms startup preparation and 50ms seek original-preview threshold.
- Fixes diagnostic scroll / selection resets during live updates; 360 real WPF refresh checks plus UI self-tests.
- Adds a dedicated offline engine builder with the existing precision policy; full SDK developer executables are not bundled.
- Includes Korean and English offline HTML manuals using three user-supplied RIFE screenshots.

Known issues: remaining startup drops; VSR contention at high output rates; other-GPU validation outstanding; no HDR10+ dynamic metadata forwarding; Dolby Vision profile/FEL/display coverage is not complete.
