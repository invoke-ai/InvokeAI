import type { AlertStatus } from '@invoke-ai/ui-library';
import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { extractMessageFromAssertionError } from 'common/util/extractMessageFromAssertionError';
import { withResult, withResultAsync } from 'common/util/result';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildRunwayVideoGraph } from 'features/nodes/util/graph/generation/buildRunwayVideoGraph';
import { buildVeo3VideoGraph } from 'features/nodes/util/graph/generation/buildVeo3VideoGraph';
import { selectCanvasDestination } from 'features/nodes/util/graph/graphBuilderUtils';
import type { GraphBuilderArg } from 'features/nodes/util/graph/types';
import { UnsupportedGenerationModeError } from 'features/nodes/util/graph/types';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import { assert, AssertionError } from 'tsafe';

const log = logger('generation');
export const enqueueRequestedVideos = createAction('app/enqueueRequestedVideos');

const enqueueVideo = async (store: AppStore, prepend: boolean) => {
  const { dispatch, getState } = store;

  dispatch(enqueueRequestedVideos());

  const state = getState();

  const model = state.video.videoModel;
  if (!model) {
    log.error('No model found in state');
    return;
  }
  const base = model.base;

  const buildGraphResult = await withResultAsync(async () => {
    const graphBuilderArg: GraphBuilderArg = { generationMode: 'txt2img', state, manager: null };

    switch (base) {
      case 'veo3':
        return await buildVeo3VideoGraph(graphBuilderArg);
      case 'runway':
        return await buildRunwayVideoGraph(graphBuilderArg);
      default:
        assert(false, `No graph builders for base ${base}`);
    }
  });

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
    toast({
      status,
      title,
      description,
    });
    return;
  }

  const { g, positivePrompt, seed } = buildGraphResult.value;

  const prepareBatchResult = withResult(() =>
    prepareLinearUIBatch({
      state,
      g,
      base,
      prepend,
      seedNode: seed,
      positivePromptNode: positivePrompt,
      origin: 'videos',
      destination: 'gallery',
    })
  );

  if (prepareBatchResult.isErr()) {
    log.error({ error: serializeError(prepareBatchResult.error) }, 'Failed to prepare batch');
    return;
  }

  const batchConfig = prepareBatchResult.value;

  // const batchConfig = {
  //   prepend,
  //   batch: {
  //     graph: g.getGraph(),
  //     runs: 1,
  //     origin,
  //     destination,
  //   },
  // };

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
      ...enqueueMutationFixedCacheKeyOptions,
      track: false,
    })
  );

  const enqueueResult = await req.unwrap();

  return { batchConfig, enqueueResult };
};

export const useEnqueueVideo = () => {
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean) => {
      return enqueueVideo(store, prepend);
    },
    [store]
  );
  return enqueue;
};
