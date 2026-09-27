import type {
  QueueEnqueueGenerateRequest,
  QueueEnqueueResult,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
  QueueResultVideoOptions,
} from '@features/queue/core/types';

import {
  buildGeneratePromptBatchPlan,
  buildLegacyGeneratePromptBatchPlan,
  buildQueueWorkflowBatchPlan,
  sanitizeBatchCount,
} from '@features/queue/core/promptBatch';
import { mapWithConcurrency } from '@platform/core/concurrency';
import { addOutputImageNames } from '@platform/core/outputImages';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { normalizeServerTimestamp } from '@platform/time/serverTimestamp';
import { absolutizeApiUrl, ApiError, apiFetch, apiFetchJson } from '@platform/transport/http';

import type { QueueImageDTO, QueueServerItemDTO } from './serverTypes';

import { buildQueueItemOrigin } from './events';
import { getQueueItem } from './serverApi';

const getQueueIdempotencyKey = (projectId: string, sourceQueueItemId: string): string =>
  `webv2:${projectId}:${sourceQueueItemId}`;

export const acknowledgeQueueEnqueue = async (projectId: string, sourceQueueItemId: string): Promise<void> => {
  await apiFetch('/api/v1/queue/default/enqueue_batch/acknowledge', {
    body: JSON.stringify({ idempotency_key: getQueueIdempotencyKey(projectId, sourceQueueItemId) }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });
};

export const getQueueEnqueueReceipt = async (
  projectId: string,
  sourceQueueItemId: string
): Promise<QueueEnqueueResult | null> => {
  const query = new URLSearchParams({ idempotency_key: getQueueIdempotencyKey(projectId, sourceQueueItemId) });
  try {
    const result = await apiFetchJson<unknown>(`/api/v1/queue/default/enqueue_batch/receipt?${query}`);
    return mapEnqueueResult(result, true);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
};

const mapEnqueueResult = (value: unknown, isReceipt = false): QueueEnqueueResult => {
  const result = value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  const batch = result.batch && typeof result.batch === 'object' ? (result.batch as Record<string, unknown>) : {};
  const batchId = isReceipt ? result.batch_id : batch.batch_id;
  const { enqueued, item_ids: itemIds, requested } = result;
  if (
    typeof batchId !== 'string' ||
    batchId.length === 0 ||
    typeof requested !== 'number' ||
    !Number.isSafeInteger(requested) ||
    requested < 1 ||
    typeof enqueued !== 'number' ||
    !Number.isSafeInteger(enqueued) ||
    enqueued < (isReceipt ? 1 : 0) ||
    enqueued > requested ||
    !Array.isArray(itemIds) ||
    itemIds.length !== enqueued ||
    new Set(itemIds).size !== itemIds.length ||
    itemIds.some((id) => !Number.isSafeInteger(id) || id <= 0)
  ) {
    throw new Error('Invalid queue enqueue response');
  }
  return { batchId, enqueued, itemIds, requested };
};

export const enqueueGenerate = async (request: QueueEnqueueGenerateRequest): Promise<QueueEnqueueResult> => {
  const planInput = {
    batchCount: sanitizeBatchCount(request.batchCount),
    negativePrompt: request.negativePrompt,
    negativePromptNodeId: request.negativePromptNodeId,
    positivePromptNodeId: request.positivePromptNodeId,
    prompts: request.positivePrompts?.length ? request.positivePrompts : [request.positivePrompt],
    seed: request.seed,
    seedBehaviour: request.seedBehaviour ?? 'per-iteration',
    seedNodeId: request.seedNodeId,
  };
  const plan = (request.legacySeedPlan ? buildLegacyGeneratePromptBatchPlan : buildGeneratePromptBatchPlan)({
    ...planInput,
    seedStep: request.seedStep,
  });
  const result = await apiFetchJson<unknown>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({
      batch: {
        data: plan.data,
        destination: request.destination,
        graph: request.graph,
        idempotency_key: getQueueIdempotencyKey(request.projectId, request.sourceQueueItemId),
        project_id: request.projectId,
        origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
        runs: plan.runs,
      },
      prepend: false,
    }),
    method: 'POST',
  });

  return mapEnqueueResult(result);
};

export const enqueueWorkflow = async (request: QueueEnqueueWorkflowRequest): Promise<QueueEnqueueResult> => {
  const plan = buildQueueWorkflowBatchPlan({
    batchCount: request.batchCount,
    batchData: request.batchData,
    seeds: request.seeds,
  });
  const result = await apiFetchJson<unknown>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({
      batch: {
        ...(plan.data ? { data: plan.data } : {}),
        destination: request.destination,
        graph: request.graph,
        idempotency_key: getQueueIdempotencyKey(request.projectId, request.sourceQueueItemId),
        project_id: request.projectId,
        origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
        runs: plan.runs,
        ...(request.workflow ? { workflow: request.workflow } : {}),
      },
      prepend: false,
    }),
    method: 'POST',
  });

  return mapEnqueueResult(result);
};

export const enqueueUtility = async (request: {
  graph: QueueEnqueueWorkflowRequest['graph'];
  origin: string;
}): Promise<{ enqueued: number; itemIds: number[] }> => {
  const result = await apiFetchJson<{ enqueued?: number; item_ids?: number[] }>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({ batch: { graph: request.graph, origin: request.origin, runs: 1 }, prepend: false }),
    method: 'POST',
  });

  return { enqueued: result.enqueued ?? 0, itemIds: result.item_ids ?? [] };
};

const getResultImageNames = (queueItem: QueueServerItemDTO, options?: QueueResultImageOptions): string[] => {
  const results = queueItem.session?.results ?? {};
  const preparedSourceMapping = queueItem.session?.prepared_source_mapping ?? {};
  const resultValues = options?.resultNodeIds
    ? Object.entries(results)
        .filter(([nodeId]) => options.resultNodeIds?.includes(preparedSourceMapping[nodeId] ?? nodeId))
        .map(([, result]) => result)
    : Object.values(results);

  const imageNames = new Set<string>();
  for (const result of resultValues) {
    addOutputImageNames(result, imageNames);
  }
  return [...imageNames];
};

const getResultImage = async (
  imageName: string,
  queuedAt: string,
  sourceQueueItemId: string,
  signal: AbortSignal
): Promise<QueueResultImage | null> => {
  try {
    const image = await apiFetchJson<QueueImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`, { signal });

    return {
      ...(image.board_id ? { boardId: image.board_id } : {}),
      createdAt: normalizeServerTimestamp(image.created_at),
      height: image.height,
      imageName: image.image_name,
      imageUrl: absolutizeApiUrl(image.image_url),
      isIntermediate: image.is_intermediate,
      queuedAt,
      sourceQueueItemId,
      thumbnailUrl: absolutizeApiUrl(image.thumbnail_url),
      width: image.width,
    };
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }
    throw error;
  }
};

export const getResultImages = async (
  itemId: number,
  sourceQueueItemId: string,
  queuedAt: string,
  options?: QueueResultImageOptions
): Promise<QueueResultImage[]> => {
  const owner = captureAccountScope();
  const item = await getQueueItem(itemId, owner.signal);

  assertAccountScopeCurrent(owner);
  const images = await mapWithConcurrency(
    getResultImageNames(item, options),
    8,
    (imageName) => getResultImage(imageName, queuedAt, sourceQueueItemId, owner.signal),
    { signal: owner.signal }
  );

  assertAccountScopeCurrent(owner);
  return images.filter((image): image is QueueResultImage => image !== null);
};

const collectResultVideoNames = (queueItem: QueueServerItemDTO, options?: QueueResultImageOptions): string[] => {
  const videoNames = new Set<string>();
  const results = queueItem.session?.results ?? {};
  const preparedSourceMapping = queueItem.session?.prepared_source_mapping ?? {};
  const resultValues = options?.resultNodeIds
    ? Object.entries(results)
        .filter(([nodeId]) => options.resultNodeIds?.includes(preparedSourceMapping[nodeId] ?? nodeId))
        .map(([, result]) => result)
    : Object.values(results);

  for (const result of resultValues) {
    if (!result || typeof result !== 'object') {
      continue;
    }

    // VideoOutput shape: { video: { video_name }, width, height, ... }.
    const videoName = (result as { video?: { video_name?: unknown } }).video?.video_name;
    if (typeof videoName === 'string') {
      videoNames.add(videoName);
    }
  }

  return [...videoNames];
};

/**
 * Treat transport errors as non-intermediate for best-effort attachment; gallery listings still hide actual
 * intermediates.
 */
const isIntermediateVideo = async (videoName: string, signal: AbortSignal): Promise<boolean> => {
  try {
    const video = await apiFetchJson<{ is_intermediate?: unknown }>(
      `/api/v1/videos/i/${encodeURIComponent(videoName)}`,
      { signal }
    );

    return video.is_intermediate === true;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      // The video is already gone; report it intermediate so the caller drops it.
      return true;
    }
    return false;
  }
};

/** Fetch result video names; hydrate DTOs only when filtering intermediates requires their flags. */
export const getResultVideoNames = async (itemId: number, options?: QueueResultVideoOptions): Promise<string[]> => {
  const owner = captureAccountScope();
  const item = await getQueueItem(itemId, owner.signal);

  assertAccountScopeCurrent(owner);
  const videoNames = collectResultVideoNames(item, options);

  if (!options?.excludeIntermediate || videoNames.length === 0) {
    return videoNames;
  }

  const intermediateFlags = await mapWithConcurrency(videoNames, 8, (name) => isIntermediateVideo(name, owner.signal), {
    signal: owner.signal,
  });

  assertAccountScopeCurrent(owner);
  return videoNames.filter((_, index) => !intermediateFlags[index]);
};
