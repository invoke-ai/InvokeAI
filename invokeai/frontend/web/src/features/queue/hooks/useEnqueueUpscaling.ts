import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { useAppStore } from 'app/store/storeHooks';
import { positivePromptAddedToHistory, selectPositivePrompt } from 'features/controlLayers/store/paramsSlice';
import { refreshDynamicPromptsForEnqueue } from 'features/dynamicPrompts/util/refreshDynamicPromptsForEnqueue';
import type { BaseModelType } from 'features/nodes/types/common';
import { prepareLinearUIBatch } from 'features/nodes/util/graph/buildLinearBatchConfig';
import { buildMultidiffusionUpscaleGraph } from 'features/nodes/util/graph/buildMultidiffusionUpscaleGraph';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';

const log = logger('generation');

const enqueueUpscaling = async (store: AppStore, prepend: boolean, dynamicPromptsErrorTitle: string) => {
  const { dispatch } = store;

  const state = await refreshDynamicPromptsForEnqueue(store);
  if (!state) {
    toast({
      status: 'error',
      title: dynamicPromptsErrorTitle,
    });
    return;
  }

  const model = state.params.model;
  if (!model) {
    log.error('No model found in state');
    return;
  }
  const base = model.base;

  const { g, seed, positivePrompt } = await buildMultidiffusionUpscaleGraph(state);

  const batchConfig = prepareLinearUIBatch({
    state,
    g,
    base: base as BaseModelType,
    prepend,
    seedNode: seed,
    positivePromptNode: positivePrompt,
    origin: 'upscaling',
    destination: 'gallery',
  });

  const req = dispatch(
    queueApi.endpoints.enqueueBatch.initiate(batchConfig, { ...enqueueMutationFixedCacheKeyOptions, track: false })
  );
  const enqueueResult = await req.unwrap();

  // Push to prompt history on successful enqueue
  dispatch(positivePromptAddedToHistory(selectPositivePrompt(state)));

  return { batchConfig, enqueueResult };
};

export const useEnqueueUpscaling = () => {
  const { t } = useTranslation();
  const store = useAppStore();
  const enqueue = useCallback(
    (prepend: boolean) => {
      return enqueueUpscaling(store, prepend, t('dynamicPrompts.resolveFailed'));
    },
    [store, t]
  );
  return enqueue;
};
