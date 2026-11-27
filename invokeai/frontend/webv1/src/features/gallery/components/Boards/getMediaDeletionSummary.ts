type MediaDeletionResult = {
  failed_images?: readonly string[];
  failed_videos?: readonly string[];
};

export const getMediaDeletionSummary = (results: PromiseSettledResult<MediaDeletionResult>[]) => ({
  failedCount: results.reduce(
    (count, result) =>
      result.status === 'fulfilled'
        ? count + (result.value.failed_images?.length ?? 0) + (result.value.failed_videos?.length ?? 0)
        : count,
    0
  ),
  requestFailed: results.some((result) => result.status === 'rejected'),
});
