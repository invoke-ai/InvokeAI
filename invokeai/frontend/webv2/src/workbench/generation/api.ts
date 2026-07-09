import type { BackendGraphContract } from '@workbench/types';

import { buildQueueItemOrigin } from '@workbench/backend/events';
import { absolutizeApiUrl, ApiError, apiFetchJson } from '@workbench/backend/http';

import type {
  EnqueueGenerateRequest,
  EnqueueGenerateResult,
  EnqueueWorkflowRequest,
  ImageDTO,
  MainModelConfig,
  QueueItemDTO,
} from './types';

import { sanitizeBatchCount } from './batch';
import { generateSeedSequence } from './graph';

export const listMainModels = async (): Promise<MainModelConfig[]> => {
  const body = await apiFetchJson<{ models?: MainModelConfig[] }>('/api/v2/models/?model_type=main');

  return body.models ?? [];
};

export const enqueueGenerateGraph = async (request: EnqueueGenerateRequest): Promise<EnqueueGenerateResult> => {
  const batchCount = sanitizeBatchCount(request.batchCount);
  const seeds = request.shouldRandomizeSeed ? generateSeedSequence(request.seed, batchCount) : [request.seed];
  const prompts = request.shouldRandomizeSeed ? seeds.map(() => request.positivePrompt) : [request.positivePrompt];
  const negativePrompts = request.shouldRandomizeSeed
    ? seeds.map(() => request.negativePrompt)
    : [request.negativePrompt];
  const body = {
    batch: {
      data: [
        [
          { field_name: 'value', items: seeds, node_path: request.seedNodeId },
          { field_name: 'value', items: prompts, node_path: request.positivePromptNodeId },
          { field_name: 'value', items: negativePrompts, node_path: request.negativePromptNodeId },
        ],
      ],
      destination: request.destination,
      graph: request.graph satisfies BackendGraphContract,
      origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
      runs: request.shouldRandomizeSeed ? 1 : batchCount,
    },
    prepend: false,
  };
  const result = await apiFetchJson<{
    batch?: { batch_id?: string };
    enqueued?: number;
    item_ids?: number[];
    requested?: number;
  }>('/api/v1/queue/default/enqueue_batch', { body: JSON.stringify(body), method: 'POST' });

  return {
    batchId: result.batch?.batch_id,
    enqueued: result.enqueued ?? 0,
    itemIds: result.item_ids ?? [],
    requested: result.requested ?? 0,
  };
};

/** Enqueue an arbitrary compiled graph — the workflow path. */
export const enqueueWorkflowGraph = async (request: EnqueueWorkflowRequest): Promise<EnqueueGenerateResult> => {
  const batchCount = sanitizeBatchCount(request.batchCount);
  const body = {
    batch: {
      destination: request.destination,
      graph: request.graph satisfies BackendGraphContract,
      origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
      runs: batchCount,
    },
    prepend: false,
  };
  const result = await apiFetchJson<{
    batch?: { batch_id?: string };
    enqueued?: number;
    item_ids?: number[];
    requested?: number;
  }>('/api/v1/queue/default/enqueue_batch', { body: JSON.stringify(body), method: 'POST' });

  return {
    batchId: result.batch?.batch_id,
    enqueued: result.enqueued ?? 0,
    itemIds: result.item_ids ?? [],
    requested: result.requested ?? 0,
  };
};

/**
 * Enqueues a small graph OUTSIDE any project's queue, tagged with a caller-built
 * utility origin (`webv2:util:<id>`). Used by {@link import('@workbench/canvas-engine/backend/utilityQueue').runUtilityGraph}
 * for filter previews / SAM: the origin is intentionally opaque to project
 * routing (see `events.ts`), so results are never staged or added to the gallery.
 * A single deterministic run (no seed/prompt batch data).
 */
export const enqueueUtilityGraph = async (request: {
  graph: BackendGraphContract;
  origin: string;
}): Promise<{ itemIds: number[]; enqueued: number }> => {
  const body = {
    batch: {
      graph: request.graph satisfies BackendGraphContract,
      origin: request.origin,
      runs: 1,
    },
    prepend: false,
  };
  const result = await apiFetchJson<{ enqueued?: number; item_ids?: number[] }>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify(body),
    method: 'POST',
  });

  return { enqueued: result.enqueued ?? 0, itemIds: result.item_ids ?? [] };
};

export const getQueueItem = (itemId: number): Promise<QueueItemDTO> =>
  apiFetchJson<QueueItemDTO>(`/api/v1/queue/default/i/${itemId}`);

export const listAllQueueItems = (): Promise<QueueItemDTO[]> =>
  apiFetchJson<QueueItemDTO[]>('/api/v1/queue/default/list_all');

export interface QueueItemResultImageOptions {
  /** Source graph node ids whose prepared execution results should be read. */
  resultNodeIds?: readonly string[];
}

const getImageDTO = async (
  imageName: string,
  queuedAt: string,
  sourceQueueItemId: string
): Promise<ImageDTO | null> => {
  try {
    const body = await apiFetchJson<{
      image_name: string;
      image_url: string;
      thumbnail_url: string;
      width: number;
      height: number;
      is_intermediate: boolean;
    }>(`/api/v1/images/i/${encodeURIComponent(imageName)}`);

    return {
      height: body.height,
      imageName: body.image_name,
      imageUrl: absolutizeApiUrl(body.image_url),
      isIntermediate: body.is_intermediate,
      queuedAt,
      sourceQueueItemId,
      thumbnailUrl: absolutizeApiUrl(body.thumbnail_url),
      width: body.width,
    };
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null;
    }

    throw error;
  }
};

const getResultImageNames = (queueItem: QueueItemDTO, options?: QueueItemResultImageOptions): string[] => {
  const imageNames = new Set<string>();
  const results = queueItem.session?.results ?? {};
  const preparedSourceMapping = queueItem.session?.prepared_source_mapping ?? {};
  const resultNodeIds = options?.resultNodeIds;
  const resultValues = resultNodeIds
    ? Object.entries(results)
        // Backend result keys are prepared execution ids; source ids carry the graph contract.
        .filter(([nodeId]) => resultNodeIds.includes(preparedSourceMapping[nodeId] ?? nodeId))
        .map(([, result]) => result)
    : Object.values(results);

  for (const result of resultValues) {
    if (!result || typeof result !== 'object') {
      continue;
    }

    const image = (result as { image?: { image_name?: unknown } }).image;
    if (typeof image?.image_name === 'string') {
      imageNames.add(image.image_name);
    }

    const collection = (result as { collection?: unknown }).collection;
    if (Array.isArray(collection)) {
      for (const item of collection) {
        if (!item || typeof item !== 'object') {
          continue;
        }

        const imageName = (item as { image_name?: unknown }).image_name;
        if (typeof imageName === 'string') {
          imageNames.add(imageName);
        }
      }
    }
  }

  return [...imageNames];
};

/** Fetch the result image DTOs of a completed queue item. */
export const getQueueItemResultImages = async (
  itemId: number,
  sourceQueueItemId: string,
  queuedAt: string,
  options?: QueueItemResultImageOptions
): Promise<ImageDTO[]> => {
  const queueItem = await getQueueItem(itemId);

  const images = await Promise.all(
    getResultImageNames(queueItem, options).map((imageName) => getImageDTO(imageName, queuedAt, sourceQueueItemId))
  );

  return images.filter((image): image is ImageDTO => image !== null);
};

export const cancelQueueItemsByBatchIds = async (batchIds: string[]): Promise<void> => {
  if (batchIds.length === 0) {
    return;
  }

  await apiFetchJson('/api/v1/queue/default/cancel_by_batch_ids', {
    body: JSON.stringify({ batch_ids: batchIds }),
    method: 'PUT',
  });
};

export const getCurrentQueueItem = (): Promise<QueueItemDTO | null> =>
  apiFetchJson<QueueItemDTO | null>('/api/v1/queue/default/current');

export const cancelQueueItem = (itemId: number): Promise<QueueItemDTO> =>
  apiFetchJson<QueueItemDTO>(`/api/v1/queue/default/i/${itemId}/cancel`, { method: 'PUT' });

export const cancelQueueItems = async (itemIds: number[]): Promise<void> => {
  await Promise.all(itemIds.map(cancelQueueItem));
};

export const cancelCurrentQueueItem = async (): Promise<QueueItemDTO | null> => {
  const currentQueueItem = await getCurrentQueueItem();

  if (!currentQueueItem) {
    return null;
  }

  return cancelQueueItem(currentQueueItem.item_id);
};

export const cancelAllExceptCurrentQueueItems = (): Promise<unknown> =>
  apiFetchJson('/api/v1/queue/default/cancel_all_except_current', { method: 'PUT' });

export const resumeQueueProcessor = (): Promise<unknown> =>
  apiFetchJson('/api/v1/queue/default/processor/resume', { method: 'PUT' });

export const pauseQueueProcessor = (): Promise<unknown> =>
  apiFetchJson('/api/v1/queue/default/processor/pause', { method: 'PUT' });
