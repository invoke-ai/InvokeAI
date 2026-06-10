import { absolutizeApiUrl, apiFetchJson } from '../backend/http';
import { buildQueueItemOrigin } from '../backend/events';
import type { BackendGraphContract } from '../types';
import type { EnqueueGenerateRequest, EnqueueGenerateResult, ImageDTO, MainModelConfig, QueueItemDTO } from './types';

export const listMainModels = async (): Promise<MainModelConfig[]> => {
  const body = await apiFetchJson<{ models?: MainModelConfig[] }>('/api/v2/models/?model_type=main');

  return body.models ?? [];
};

export const enqueueGenerateGraph = async (request: EnqueueGenerateRequest): Promise<EnqueueGenerateResult> => {
  const body = {
    batch: {
      data: [
        [
          { field_name: 'value', items: [request.seed], node_path: request.seedNodeId },
          { field_name: 'value', items: [request.positivePrompt], node_path: request.positivePromptNodeId },
          { field_name: 'value', items: [request.negativePrompt], node_path: request.negativePromptNodeId },
        ],
      ],
      destination: request.destination,
      graph: request.graph satisfies BackendGraphContract,
      origin: buildQueueItemOrigin(request.sourceQueueItemId),
      runs: request.batchCount,
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

const getImageDTO = async (imageName: string, queuedAt: string, sourceQueueItemId: string): Promise<ImageDTO> => {
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

  return Promise.all(
    getResultImageNames(queueItem).map((imageName) => getImageDTO(imageName, queuedAt, sourceQueueItemId))
  );
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
