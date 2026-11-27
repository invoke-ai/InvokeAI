export const MIN_BATCH_COUNT = 1;

export const sanitizeBatchCount = (batchCount: unknown): number =>
  typeof batchCount === 'number' && Number.isFinite(batchCount)
    ? Math.max(MIN_BATCH_COUNT, Math.round(batchCount))
    : MIN_BATCH_COUNT;
