# Canvas pixel goldens

Reviewed image baselines for the v2 compositor, produced by the Chromium raster suite.

- `__golden__/<name>.png` is the reviewed frame; `__golden__/<name>.json` records the browser,
  device pixel ratio, size, flatten background and tolerance it was produced with.
- Frames are flattened over an opaque background before comparison and stored as PNG, so the
  comparison is exact for integer operations. Interpolation-sensitive frames (rotation, scaling)
  declare a per-channel and differing-pixel tolerance in the test and in the record.
- A failing comparison writes `expected`, `actual` and `diff` PNGs under the gitignored
  `__screenshots__/golden/` directory next to this file.

Update baselines only after reviewing the artifacts:

```sh
pnpm run test:browser:update-goldens
```

The command rewrites every golden the suite renders; commit the PNG and JSON pairs together and
describe the rendering change in the pull request.
