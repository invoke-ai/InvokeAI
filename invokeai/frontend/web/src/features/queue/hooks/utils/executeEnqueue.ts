import type { ActionCreatorWithoutPayload } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStore } from 'app/store/store';
import { serializeError } from 'serialize-error';
import { enqueueMutationFixedCacheKeyOptions, queueApi } from 'services/api/endpoints/queue';
import type { paths } from 'services/api/schema';

export type EnqueueBatchArg =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['requestBody']['content']['application/json'];
export type EnqueueBatchResponse =
  paths['/api/v1/queue/{queue_id}/enqueue_batch']['post']['responses']['201']['content']['application/json'];

export type EnqueueOptionsBase = { prepend: boolean };

interface ExecuteEnqueueConfig<TOptions extends EnqueueOptionsBase, TBuildResult> {
  store: AppStore;
  options: TOptions;
  requestedAction: ActionCreatorWithoutPayload<string>;
  build: (context: { store: AppStore; options: TOptions }) => Promise<TBuildResult | null>;
  prepareBatch: (context: { store: AppStore; options: TOptions; buildResult: TBuildResult }) => EnqueueBatchArg;
  onSuccess?: (context: {
    store: AppStore;
    options: TOptions;
    buildResult: TBuildResult;
    batch: EnqueueBatchArg;
    response: EnqueueBatchResponse;
  }) => void;
  onError?: (context: { store: AppStore; options: TOptions; error: unknown }) => void;
  log?: ReturnType<typeof logger>;
}

export const executeEnqueue = async <TOptions extends EnqueueOptionsBase, TBuildResult>({
  store,
  options,
  requestedAction,
  build,
  prepareBatch,
  onSuccess,
  onError,
  log = logger('enqueue'),
}: ExecuteEnqueueConfig<TOptions, TBuildResult>) => {
  const { dispatch } = store;
  dispatch(requestedAction());

  try {
    const buildResult = await build({ store, options });
    if (!buildResult) {
      return null;
    }

    const batchConfig = prepareBatch({ store, options, buildResult });

    const req = dispatch(
      queueApi.endpoints.enqueueBatch.initiate(batchConfig, {
        ...enqueueMutationFixedCacheKeyOptions,
        track: false,
      })
    );

    const enqueueResult = await req.unwrap();

    onSuccess?.({ store, options, buildResult, batch: batchConfig, response: enqueueResult });

    return { batchConfig, enqueueResult };
  } catch (error) {
    log.error({ error: serializeError(error as Error) }, 'Failed to enqueue batch');
    onError?.({ store, options, error });
    return null;
  }
};
