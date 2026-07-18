import type { QueueItem } from './types';

import { sanitizeBatchCount } from './generation/batch';
import { getUpscaleOutputDimensions, normalizeUpscaleWidgetValues } from './upscale/settings';

/** Values owned by the source that created this immutable queue snapshot. */
export const getQueueItemSourceWidgetValues = (item: QueueItem): Record<string, unknown> => {
  if (item.snapshot.sourceId === 'upscale') {
    return item.snapshot.widgetStates.upscale?.values ?? {};
  }

  // Canvas intentionally shares Generate settings. Workflow has no first-party
  // batch form yet and keeps the historical Generate batch-count fallback.
  return item.snapshot.widgetStates.generate?.values ?? {};
};

export const getQueueItemSnapshotBatchCount = (item: QueueItem): number =>
  sanitizeBatchCount(getQueueItemSourceWidgetValues(item).batchCount);

export const getQueueItemSnapshotDimensions = (
  item: QueueItem,
  fallback: { width: number; height: number }
): { width: number; height: number } => {
  if (item.snapshot.sourceId === 'upscale') {
    const values = normalizeUpscaleWidgetValues(item.snapshot.widgetStates.upscale?.values);

    if (values?.inputImage) {
      return getUpscaleOutputDimensions(values.inputImage, values.scale);
    }
  }

  const values = getQueueItemSourceWidgetValues(item);
  const width = typeof values.width === 'number' && Number.isFinite(values.width) ? values.width : fallback.width;
  const height = typeof values.height === 'number' && Number.isFinite(values.height) ? values.height : fallback.height;

  return { height: Math.max(1, Math.round(height)), width: Math.max(1, Math.round(width)) };
};
