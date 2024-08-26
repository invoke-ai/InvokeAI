import { logger } from 'app/logging/logger';
import { enqueueRequested } from 'app/store/actions';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import type { SerializableObject } from 'common/types';
import type { Result } from 'common/util/result';
import { isErr, withResult, withResultAsync } from 'common/util/result';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { sessionStagingAreaReset, sessionStartedStaging } from 'features/controlLayers/store/canvasV2Slice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildSD1Graph } from 'features/nodes/util/graph/generation/buildSD1Graph';
import { buildSDXLGraph } from 'features/nodes/util/graph/generation/buildSDXLGraph';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { serializeError } from 'serialize-error';
import { queueApi } from 'services/api/endpoints/queue';
import type { Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('generation');

export const addEnqueueRequestedLinear = (startAppListening: AppStartListening) => {
  startAppListening({
    predicate: (action): action is ReturnType<typeof enqueueRequested> =>
      enqueueRequested.match(action) && action.payload.tabName === 'generation',
    effect: async (action, { getState, dispatch }) => {
      const state = getState();
      const model = state.params.model;
      const { prepend } = action.payload;

      const manager = $canvasManager.get();
      assert(manager, 'No model found in state');

      let didStartStaging = false;

      if (!state.canvasV2.session.isStaging && state.canvasV2.session.mode === 'compose') {
        dispatch(sessionStartedStaging());
        didStartStaging = true;
      }

      const abortStaging = () => {
        if (didStartStaging && getState().canvasV2.session.isStaging) {
          dispatch(sessionStagingAreaReset());
        }
      };

      let buildGraphResult: Result<
        { g: Graph; noise: Invocation<'noise'>; posCond: Invocation<'compel' | 'sdxl_compel_prompt'> },
        Error
      >;

      assert(model, 'No model found in state');
      const base = model.base;

      switch (base) {
        case 'sdxl':
          buildGraphResult = await withResultAsync(() => buildSDXLGraph(state, manager));
          break;
        case 'sd-1':
        case `sd-2`:
          buildGraphResult = await withResultAsync(() => buildSD1Graph(state, manager));
          break;
        default:
          assert(false, `No graph builders for base ${base}`);
      }

      if (isErr(buildGraphResult)) {
        log.error({ error: serializeError(buildGraphResult.error) }, 'Failed to build graph');
        abortStaging();
        return;
      }

      const { g, noise, posCond } = buildGraphResult.value;

      const prepareBatchResult = withResult(() => prepareLinearUIBatch(state, g, prepend, noise, posCond));

      if (isErr(prepareBatchResult)) {
        log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
        abortStaging();
        return;
      }

      const req = dispatch(
        queueApi.endpoints.enqueueBatch.initiate(prepareBatchResult.value, {
          fixedCacheKey: 'enqueueBatch',
        })
      );
      req.reset();

      const enqueueResult = await withResultAsync(() => req.unwrap());

      if (isErr(enqueueResult)) {
        log.error({ error: serializeError(enqueueResult.error) }, 'Failed to enqueue batch');
        abortStaging();
        return;
      }

      log.debug({ batchConfig: prepareBatchResult.value } as SerializableObject, 'Enqueued batch');
    },
  });
};
