# Inference and model backend

## Design and resources

- Trace callers, loaders/caches, tensor shapes, device/dtype handling, and cleanup before inference changes. Reusable model/inference policy belongs here, not UI/API adapters.
- Use existing device/model-management abstractions, including `util/devices.py`; preserve CPU/MPS/ROCm/XPU/multi-device support without hardcoded CUDA, GPU zero, or dtype assumptions.
- Respect session/thread device ownership. Restore temporary model patches, device state, hooks, and allocations on success/cancellation/exceptions; avoid request-specific mutable globals.
- Define model/cache lifetimes and working-memory estimates. Bound queues, caches, downloads, batches, and buffers; consider peak memory and throughput.
- Preserve model validation/loading safeguards, vendored exclusions, and upstream provenance; avoid unrelated vendored rewrites/formatting.

## Performance

- Inspect duplicate work, device transfers, dtype conversions, tensor copies, synchronization, repeated loading, and retained tensors.
- Measure material latency/throughput/peak-memory changes with representative shapes, resolutions, batches, and devices; distinguish cold loading/compilation from warm inference.
- Preserve output semantics, cancellation, and correctness at larger inputs, using appropriate numerical tolerances and reproducible seeds where supported.
- Kernels, caches, compilation, or concurrency need demonstrated benefit and clear fallback/lifecycle behavior.

## Verification

- Read `tests/AGENTS.md` and relevant suites in `tests/backend/`, `tests/model_identification/`, or `tests/test_model_manager/`.
- Use small CPU fixtures and targeted device checks; real tensor operations must expose shape/dtype/numerical/memory errors mocks would hide.
- Cover affected failure cleanup, unsupported configurations, boundary shapes, and cancellation. No routine large-model downloads or user model/output directories for unit tests.
- Run focused pytest and root Ruff. For material inference changes, report hardware, dtype, shapes, measurements, and untested platforms/model paths; CPU passes do not prove accelerator behavior.
