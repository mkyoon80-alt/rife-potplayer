# RIFE control application

WPF / .NET 10 self-contained Windows x64 application. Build with `build-ui.ps1`; the SDK version is pinned in `dotnet-sdk.json`. Runtime, filter and model payloads belong beside the published app as described in the main manual.

`--self-test` checks settings, layout and status handling. `--test-diagnostics-scroll <report.json>` uses actual offscreen WPF controls and dispatcher turns to reproduce the old TextBox reset and verify scroll, selection, focus, subsequent input, tab switching, short and empty reports. It does not replay a physical mouse. `--render-preview <file.png> --preview-state active|setup|expanded` creates synthetic UI-state previews, explicitly labeled as previews.

The visual identity inherits the NVOF control application's established WPF design. This release applies UI UX Pro Max and Impeccable to review the inherited UI, fix diagnostic reading state and create the bilingual manual. The three manual pictures are user-supplied RIFE screenshots, not NVOF captures. No shadcn/21st React code is required by this native app.
