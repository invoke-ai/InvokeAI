import type { AlertStatus } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import { useAppStore } from 'app/store/storeHooks';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildRunwayVideoGraph } from 'features/nodes/util/graph/generation/buildRunwayVideoGraph';
import { buildVeo3VideoGraph } from 'features/nodes/util/graph/generation/buildVeo3VideoGraph';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { AssertionError } from 'tsafe';

import type { EnqueueBatchArg } from './utils/executeEnqueue';
import { executeEnqueue } from './utils/executeEnqueue';

const log = logger('generation');
export const enqueueRequestedVideos = createAction('app/enqueueRequestedVideos');

type VideoBuildResult = {
  batchConfig: EnqueueBatchArg;
};

const getVideoGraphBuilder = (base: string) => {
  switch (base) {
    case 'veo3':
      return buildVeo3VideoGraph;
    case 'runway':
      return buildRunwayVideoGraph;
    default:
      return null;
  }
};

export const useEnqueueVideo = () => {
  const store = useAppStore();

  const enqueue = useCallback(
    (prepend: boolean) => {
      return executeEnqueue({
        store,
        options: { prepend },
        requestedAction: enqueueRequestedVideos,
        log,
        build: async ({ store: innerStore, options }) => {
          const state = innerStore.getState();

          const model = state.video.videoModel;
          if (!model) {
            log.error('No model found in state');
            return null;
          }

          const builder = getVideoGraphBuilder(model.base);
          if (!builder) {
            log.error({ base: model.base }, 'No graph builders for base');
            return null;
          }

          const graphBuilderArg: GraphBuilderArg = { generationMode: 'txt2img', state, manager: null };
          const buildGraphResult = await withResultAsync(async () => await builder(graphBuilderArg));

          if (buildGraphResult.isErr()) {
            let title = 'Failed to build graph';
            let status: AlertStatus = 'error';
            let description: string | null = null;
            if (buildGraphResult.error instanceof AssertionError) {
              description = extractMessageFromAssertionError(buildGraphResult.error);
            } else if (buildGraphResult.error instanceof UnsupportedGenerationModeError) {
              title = 'Unsupported generation mode';
              description = buildGraphResult.error.message;
              status = 'warning';
            }
            const error = serializeError(buildGraphResult.error);
            log.error({ error }, 'Failed to build graph');
            toast({ status, title, description });
            return null;
          }

          const { g, positivePrompt, seed } = buildGraphResult.value;

          const prepareBatchResult = withResult(() =>
            prepareLinearUIBatch({
              state,
              g,
              base: model.base,
              prepend: options.prepend,
              seedNode: seed,
              positivePromptNode: positivePrompt,
              origin: 'videos',
              destination: 'gallery',
            })
          );

          if (prepareBatchResult.isErr()) {
            log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
            return null;
          }

          return {
            batchConfig: prepareBatchResult.value,
          } satisfies VideoBuildResult;
        },
        prepareBatch: ({ buildResult }) => buildResult.batchConfig,
        onSuccess: ({ store: innerStore }) => {
          const state = innerStore.getState();
          innerStore.dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));
        },
      });
    },
    [store]
  );

  return enqueue;
};
