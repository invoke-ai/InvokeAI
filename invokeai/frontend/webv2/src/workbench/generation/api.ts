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

import { generateSeedSequence } from './graph';

export const listMainModels = async (): Promise<MainModelConfig[]> => {
  const body = await apiFetchJson<{ models?: MainModelConfig[] }>('/api/v2/models/?model_type=main');

  return body.models ?? [];
};

export const enqueueGenerateGraph = async (request: EnqueueGenerateRequest): Promise<EnqueueGenerateResult> => {
  const batchCount = Math.max(1, Math.round(request.batchCount));
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
      origin: buildQueueItemOrigin(request.sourceQueueItemId),
      runs: request.shouldRandomizeSeed ? 1 : batchCount,
    },
    prepend: false,
  };
  const result = await apiFetchJson<{ batch?: { batch_id?: string }; item_ids?: number[] }>(
    '/api/v1/queue/default/enqueue_batch',
    { body: JSON.stringify(body), method: 'POST' }
  );

  return { batchId: result.batch?.batch_id, itemIds: result.item_ids ?? [] };
};

/** Enqueue an arbitrary compiled graph as a single run — the workflow / project-graph path. */
export const enqueueWorkflowGraph = async (request: EnqueueWorkflowRequest): Promise<EnqueueGenerateResult> => {
  const body = {
    batch: {
      destination: request.destination,
      graph: request.graph satisfies BackendGraphContract,
      origin: buildQueueItemOrigin(request.sourceQueueItemId),
      runs: 1,
    },
    prepend: false,
  };
  const result = await apiFetchJson<{ batch?: { batch_id?: string }; item_ids?: number[] }>(
    '/api/v1/queue/default/enqueue_batch',
    { body: JSON.stringify(body), method: 'POST' }
  );

  return { batchId: result.batch?.batch_id, itemIds: result.item_ids ?? [] };
};

export const getQueueItem = (itemId: number): Promise<QueueItemDTO> =>
  apiFetchJson<QueueItemDTO>(`/api/v1/queue/default/i/${itemId}`);

export const listAllQueueItems = (): Promise<QueueItemDTO[]> =>
  apiFetchJson<QueueItemDTO[]>('/api/v1/queue/default/list_all');

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

const getResultImageNames = (queueItem: QueueItemDTO): string[] => {
  const imageNames = new Set<string>();

  for (const result of Object.values(queueItem.session?.results ?? {})) {
    if (!result || typeof result !== 'object') {
      continue;
    }

    const image = (result as { image?: { image_name?: unknown } }).image;
    if (typeof image?.image_name === 'string') {
      imageNames.add(image.image_name);
    }
  }

  return [...imageNames];
};

/** Fetch the result image DTOs of a completed queue item. */
export const getQueueItemResultImages = async (
  itemId: number,
  sourceQueueItemId: string,
  queuedAt: string
): Promise<ImageDTO[]> => {
  const queueItem = await getQueueItem(itemId);

  const images = await Promise.all(
    getResultImageNames(queueItem).map((imageName) => getImageDTO(imageName, queuedAt, sourceQueueItemId))
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

export const cancelQueueItems = async (itemIds: number[]): Promise<void> => {
  await Promise.all(
    itemIds.map((itemId) => apiFetchJson(`/api/v1/queue/default/i/${itemId}/cancel`, { method: 'PUT' }))
  );
};
