# Queue Enqueue Patterns

This directory contains the hooks and utilities that translate UI actions into queue batches. The flow is intentionally
modular so adding a new enqueue type (e.g. a new generation mode) follows a predictable recipe.

## Key building blocks

- `hooks/useEnqueue*.ts` – Feature-specific hooks (generate, canvas, upscaling, video, workflows). Each hook wires local
  state to the shared enqueue utilities.
- `hooks/utils/graphBuilders.ts` – Maps base models (sdxl, flux, etc.) to their graph builder functions and normalizes
  synchronous vs. asynchronous builders.
- `hooks/utils/executeEnqueue.ts` – Orchestrates the enqueue lifecycle:
  1. dispatch the `enqueueRequested*` action
  2. build the graph/batch data
  3. call `queueApi.endpoints.enqueueBatch`
  4. run success/error callbacks

## Adding a new enqueue type

1. **Implement the graph builder (if needed).**
   - Create the graph construction logic in `features/nodes/util/graph/generation/...` so it returns a
     `GraphBuilderReturn`.
   - If the builder reuses existing primitives, consider wiring it into `graphBuilders.ts` by extending the `graphBuilderMap`.

2. **Create the enqueue hook.**
   - Add `useEnqueue<Feature>.ts` mirroring the existing hooks. Import `executeEnqueue` and supply feature-specific
     `build`, `prepareBatch`, and `onSuccess` callbacks.
   - If the feature depends on a new base model, add it to `graphBuilders.ts`.

3. **Register the tab in `useInvoke`.**
   - `useInvoke.ts` looks up handlers based on the active tab. Import your new hook and call it inside the `switch`
     (or future registry) so the UI can enqueue from the feature.

4. **Add Redux action (optional).**
   - Most enqueue hooks dispatch a `enqueueRequested*` action for devtools visibility. Create one with `createAction` if
     you want similar tracing.

5. **Cover with tests.**
   - Unit-test feature-specific behavior (graph selection, batch tweaks). The shared helpers already have coverage in
     `hooks/utils/`.

## Tips

- Keep `build` lean: fetch state, compose graph/batch data, and return `null` when prerequisites are missing. The shared
  helper will skip enqueueing and your `onError` will handle logging.
- Use the shared `prepareLinearUIBatch` for single-graph UI workflows. For advanced cases (multi-run batches, workflow
  validation runs), supply a custom `prepareBatch` function.
- Prefer updating `graphBuilders.ts` when adding a new base model so every image-based enqueue automatically benefits.

With this structure, the main task when introducing a new enqueue type is describing how to build its graph and how to
massage the batch payload—everything else (dispatching, API calls, history updates) is handled by the utilities.
