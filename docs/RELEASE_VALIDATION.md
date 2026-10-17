# v0.3.0 release validation

The public source was rebuilt separately from the installed local trial. Playback source matches the validated candidate, except the version identity and enabling the already-tested optimization option by default. Release preparation also adds the offline optional-engine checks and preparation workflow. The live installation, settings and registration were not changed during publication.

| Release-package check | Result |
|---|---|
| Clean native build / WPF publish | Passed; native and UI version 0.3.0.0 |
| Automated tests available in public source | 31/31 passed |
| Source-size transitions / output pixels | 32 outputs matched, including 3840×1604, 8K, portrait and return to smaller sources |
| HEVC/AV1 media negotiation | 21 cases passed, including progressive, field flags, missing/invalid headers and codec mismatches |
| DirectShow startup / flush / Stop / restart | Passed with production filter at 4K HDR and AV1 8K HDR 60→240fps |
| Existing engine reuse | Full and optional plans verified; settings and plan hashes retained |
| Missing optional engines | Prepared from bundled models/resources; D3D11 pixel verification passed |
| Rebuilt-engine boundary comparison | 689 generated outputs matched the original engine |
| Preparation transaction | 7 reuse/build/failure/rollback cases passed using stubbed tools in isolated folders |
| Model extraction | Both distributed optional-model SHA256 values reproduced |
| Standalone UI / diagnostic scrolling | Self-test passed; 360 WPF refreshes preserved reading position and selection |
| Manuals | Korean/English layout, version, local image/link and offline assets checked |

The earlier candidate validation additionally covered 34 local automated tests (including private fixtures), 812 quality rows, 1,196 sequence outputs, 36 QHD/8K24×10 outputs, actual playlist transitions and rapid direction-key seeking. These counts are not added to the release tests as new independent results. Private source video, raw playback logs and personal settings are not distributed.

Two release-test setup attempts were excluded: one used a 240fps test INI while expecting a 120fps result; another used an older all-frames-interpolated timing assertion incompatible with the retained original-frame startup window. Final production-filter graph checks use the dedicated startup-window test, including timestamp continuity and cancellation. These setup failures were not product-code regressions or counted as passing checks.

Power comparison: RTX 5090, the same AV1 8K HDR 60p ×4 segment, 12 seconds × 2 runs per configuration. Average GPU board power was 210.3W with the earlier interpolation path and 200.1W with the optimizations, a 10.2W / approximately 4.9% reduction. Both configurations used the same validation build, with AV1 correction and then-current startup scheduling shared. This isolates optimization behavior; it is not a direct comparison of complete public binaries. Both measured windows had zero drops; startup drops were recorded separately.

The user separately reported 214W average GPU board power over approximately one minute with the final local playback candidate. The supplied capture shows hardware-decoded AV1 8K HDR, P010, 60p ×4 and Dropped 0. That interval is separate from the 12-second comparison. Cold startup drops at 8K60×4 remain a long-term improvement task; no universal zero-drop or fixed power-saving claim is made.
