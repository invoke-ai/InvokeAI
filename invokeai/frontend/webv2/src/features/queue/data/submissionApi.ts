import type {
  QueueEnqueueGenerateRequest,
  QueueEnqueueResult,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
  QueueResultVideoOptions,
} from '@features/queue/core/types';

import { buildGeneratePromptBatchPlan, sanitizeBatchCount } from '@features/queue/core/promptBatch';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import { absolutizeApiUrl, ApiError, apiFetchJson } from '@platform/transport/http';

import type { QueueImageDTO, QueueServerItemDTO } from './serverTypes';

import { buildQueueItemOrigin } from './events';
import { getQueueItem } from './serverApi';

const mapEnqueueResult = (result: {
  batch?: { batch_id?: string };
  enqueued?: number;
  item_ids?: number[];
  requested?: number;
}): QueueEnqueueResult => ({
  batchId: result.batch?.batch_id,
  enqueued: result.enqueued ?? 0,
  itemIds: result.item_ids ?? [],
  requested: result.requested ?? 0,
});

export const enqueueGenerate = async (request: QueueEnqueueGenerateRequest): Promise<QueueEnqueueResult> => {
  const plan = buildGeneratePromptBatchPlan({
    batchCount: sanitizeBatchCount(request.batchCount),
    negativePrompt: request.negativePrompt,
    negativePromptNodeId: request.negativePromptNodeId,
    positivePromptNodeId: request.positivePromptNodeId,
    prompts: request.positivePrompts?.length ? request.positivePrompts : [request.positivePrompt],
    seed: request.seed,
    seedBehaviour: request.seedBehaviour ?? 'per-iteration',
    seedNodeId: request.seedNodeId,
    shouldRandomizeSeed: request.shouldRandomizeSeed,
  });
  const result = await apiFetchJson<{
    batch?: { batch_id?: string };
    enqueued?: number;
    item_ids?: number[];
    requested?: number;
  }>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({
      batch: {
        data: plan.data,
        destination: request.destination,
        graph: request.graph,
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
  const result = await apiFetchJson<{
    batch?: { batch_id?: string };
    enqueued?: number;
    item_ids?: number[];
    requested?: number;
  }>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({
      batch: {
        destination: request.destination,
        graph: request.graph,
        origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
        runs: sanitizeBatchCount(request.batchCount),
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
  const imageNames = new Set<string>();
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

    const imageName = (result as { image?: { image_name?: unknown } }).image?.image_name;
    if (typeof imageName === 'string') {
      imageNames.add(imageName);
    }

    const collection = (result as { collection?: unknown }).collection;
    if (Array.isArray(collection)) {
      for (const item of collection) {
        const collectionImageName =
          item && typeof item === 'object' ? (item as { image_name?: unknown }).image_name : undefined;
        if (typeof collectionImageName === 'string') {
          imageNames.add(collectionImageName);
        }
      }
    }
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
  const images = await Promise.all(
    getResultImageNames(item, options).map((imageName) =>
      getResultImage(imageName, queuedAt, sourceQueueItemId, owner.signal)
    )
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
 * True when the video's DTO reports it as an intermediate. Fail-open on transport
 * errors: the caller's board attach is best-effort, and a wrongly-attached
 * intermediate is invisible in gallery listings (which filter intermediates).
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

/**
 * The names of the videos a completed backend item produced. The queue runtime only
 * routes them onto the destination board, so DTOs are hydrated solely when
 * `excludeIntermediate` needs the `is_intermediate` flag (the video analogue of the
 * image path's filterIntermediateResults).
 */
export const getResultVideoNames = async (itemId: number, options?: QueueResultVideoOptions): Promise<string[]> => {
  const owner = captureAccountScope();
  const item = await getQueueItem(itemId, owner.signal);

  assertAccountScopeCurrent(owner);
  const videoNames = collectResultVideoNames(item, options);

  if (!options?.excludeIntermediate || videoNames.length === 0) {
    return videoNames;
  }

  const intermediateFlags = await Promise.all(videoNames.map((name) => isIntermediateVideo(name, owner.signal)));

  assertAccountScopeCurrent(owner);
  return videoNames.filter((_, index) => !intermediateFlags[index]);
};
