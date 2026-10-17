# Inference and model backend

## Design and resources

- Before inference changes, trace callers, loaders/caches, tensor shapes, device/dtype handling, cleanup. Reusable model/inference policy lives here, not in UI/API adapters.
- Use existing device/model-management abstractions, `util/devices.py` included. Keep CPU/MPS/ROCm/XPU/multi-device support. No hardcoded CUDA, GPU zero, or dtype assumptions.
- Respect session/thread device ownership. Restore temporary model patches, device state, hooks, allocations on success, cancellation, and exceptions. No request-specific mutable globals.
- Define model/cache lifetimes and working-memory estimates. Bound queues, caches, downloads, batches, buffers. Consider peak memory and throughput.
- New or refitted working-memory coefficients go in `working_memory/data/` with the measured points that bound them; their estimators declare an `EstimatorSpec` and return a `WorkingMemoryEstimate`.
- Keep model validation/loading safeguards, vendored exclusions, upstream provenance. No unrelated vendored rewrites or formatting.

## Performance

- Check duplicate work, device transfers, dtype conversions, tensor copies, synchronization, repeated loading, retained tensors.
- Measure material latency/throughput/peak-memory changes with representative shapes, resolutions, batches, devices. Separate cold loading/compilation from warm inference.
- Keep output semantics, cancellation, correctness at larger inputs. Use fitting numerical tolerances and reproducible seeds where supported.
- Kernels, caches, compilation, concurrency need demonstrated benefit and clear fallback/lifecycle behavior.

## Verification

- Read `tests/AGENTS.md` and relevant suites in `tests/backend/`, `tests/model_identification/`, `tests/test_model_manager/`.
- Small CPU fixtures plus targeted device checks. Use real tensor operations: they expose shape/dtype/numerical/memory errors mocks hide.
- Cover affected failure cleanup, unsupported configurations, boundary shapes, cancellation. Unit tests never use user model/output directories and never routinely download large models.
- Run focused pytest and root Ruff. Material inference changes: report hardware, dtype, shapes, measurements, untested platforms/model paths. CPU passes do not prove accelerator behavior.
