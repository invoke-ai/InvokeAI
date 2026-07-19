import type {
  QueueEnqueueGenerateRequest,
  QueueEnqueueResult,
  QueueEnqueueWorkflowRequest,
  QueueResultImage,
  QueueResultImageOptions,
} from '@features/queue/core/types';

import { absolutizeApiUrl, ApiError, apiFetchJson } from '@platform/transport/http';

import type { QueueImageDTO, QueueServerItemDTO } from './serverTypes';

import { buildQueueItemOrigin } from './events';
import { getQueueItem } from './serverApi';

const SEED_MAX = 4_294_967_295;

const sanitizeBatchCount = (value: unknown): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.max(1, Math.round(value)) : 1;

const generateSeedSequence = (start: number, count: number): number[] =>
  Array.from({ length: sanitizeBatchCount(count) }, (_, index) => (start + index) % SEED_MAX);

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
  const batchCount = sanitizeBatchCount(request.batchCount);
  const seeds = request.shouldRandomizeSeed ? generateSeedSequence(request.seed, batchCount) : [request.seed];
  const prompts = request.shouldRandomizeSeed ? seeds.map(() => request.positivePrompt) : [request.positivePrompt];
  const negativePrompts = request.shouldRandomizeSeed
    ? seeds.map(() => request.negativePrompt)
    : [request.negativePrompt];
  const result = await apiFetchJson<{
    batch?: { batch_id?: string };
    enqueued?: number;
    item_ids?: number[];
    requested?: number;
  }>('/api/v1/queue/default/enqueue_batch', {
    body: JSON.stringify({
      batch: {
        data: [
          [
            { field_name: 'value', items: seeds, node_path: request.seedNodeId },
            { field_name: 'value', items: prompts, node_path: request.positivePromptNodeId },
            { field_name: 'value', items: negativePrompts, node_path: request.negativePromptNodeId },
          ],
        ],
        destination: request.destination,
        graph: request.graph,
        origin: buildQueueItemOrigin(request.sourceQueueItemId, request.projectId),
        runs: request.shouldRandomizeSeed ? 1 : batchCount,
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
  sourceQueueItemId: string
): Promise<QueueResultImage | null> => {
  try {
    const image = await apiFetchJson<QueueImageDTO>(`/api/v1/images/i/${encodeURIComponent(imageName)}`);

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
  const item = await getQueueItem(itemId);
  const images = await Promise.all(
    getResultImageNames(item, options).map((imageName) => getResultImage(imageName, queuedAt, sourceQueueItemId))
  );

  return images.filter((image): image is QueueResultImage => image !== null);
};
