# Changelog

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
