import type { BackendGraphContract } from '../types';
import type { EnqueueGenerateRequest, EnqueueGenerateResult, ImageDTO, MainModelConfig, QueueItemDTO } from './types';

const API_BASE_URL = import.meta.env.VITE_INVOKEAI_API_BASE_URL ?? '';
const POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 15 * 60 * 1000;

const buildUrl = (path: string): string => `${API_BASE_URL}${path}`;

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });

const assertOk = async (response: Response): Promise<Response> => {
  if (response.ok) {
    return response;
  }

  const text = await response.text();
  throw new Error(text || `${response.status} ${response.statusText}`);
};

const absolutizeImageUrl = (url: string): string => {
  if (!API_BASE_URL || url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }

  return new URL(url, API_BASE_URL).toString();
};

export const listMainModels = async (): Promise<MainModelConfig[]> => {
  const response = await assertOk(await fetch(buildUrl('/api/v2/models/?model_type=main')));
  const body = (await response.json()) as { models?: MainModelConfig[] };

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
      origin: `webv2:${request.sourceQueueItemId}`,
      runs: request.batchCount,
    },
    prepend: false,
  };

  const response = await assertOk(
    await fetch(buildUrl('/api/v1/queue/default/enqueue_batch'), {
      body: JSON.stringify(body),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
    })
  );
  const result = (await response.json()) as { item_ids?: number[] };

  return { itemIds: result.item_ids ?? [] };
};

const getQueueItem = async (itemId: number): Promise<QueueItemDTO> => {
  const response = await assertOk(await fetch(buildUrl(`/api/v1/queue/default/i/${itemId}`)));

  return (await response.json()) as QueueItemDTO;
};

const getImageDTO = async (imageName: string, queuedAt: string, sourceQueueItemId: string): Promise<ImageDTO> => {
  const response = await assertOk(await fetch(buildUrl(`/api/v1/images/i/${encodeURIComponent(imageName)}`)));
  const body = (await response.json()) as {
    image_name: string;
    image_url: string;
    thumbnail_url: string;
    width: number;
    height: number;
    is_intermediate: boolean;
  };

  return {
    height: body.height,
    imageName: body.image_name,
    imageUrl: absolutizeImageUrl(body.image_url),
    isIntermediate: body.is_intermediate,
    queuedAt,
    sourceQueueItemId,
    thumbnailUrl: absolutizeImageUrl(body.thumbnail_url),
    width: body.width,
  };
};

const getResultImageNames = (queueItem: QueueItemDTO): string[] => {
  const imageNames = new Set<string>();

  for (const result of Object.values(queueItem.session.results)) {
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

export const waitForQueueItemImages = async (
  itemId: number,
  sourceQueueItemId: string,
  queuedAt: string
): Promise<ImageDTO[]> => {
  const startedAt = Date.now();

  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    const queueItem = await getQueueItem(itemId);

    if (queueItem.status === 'completed') {
      const imageNames = getResultImageNames(queueItem);
      return Promise.all(imageNames.map((imageName) => getImageDTO(imageName, queuedAt, sourceQueueItemId)));
    }

    if (queueItem.status === 'failed' || queueItem.status === 'canceled') {
      throw new Error(queueItem.error_message ?? queueItem.error_type ?? `Queue item ${itemId} ${queueItem.status}.`);
    }

    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error(`Timed out waiting for queue item ${itemId}.`);
};

export const cancelQueueItems = async (itemIds: number[]): Promise<void> => {
  await Promise.all(
    itemIds.map(async (itemId) => {
      await assertOk(
        await fetch(buildUrl(`/api/v1/queue/default/i/${itemId}/cancel`), {
          method: 'PUT',
        })
      );
    })
  );
};
