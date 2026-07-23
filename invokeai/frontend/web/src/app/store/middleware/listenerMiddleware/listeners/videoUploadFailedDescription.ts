/**
 * Builds the description for the video-upload-failed toast (PR #9163 review).
 *
 * Batch video uploads are aggregated with Promise.allSettled, so a partially failed
 * batch resolves "successfully" at the call site — the per-mutation rejected listener is
 * where each failure surfaces, and it must name the file so the user can tell which of
 * their videos disappeared from a mixed-outcome batch.
 *
 * Kept in its own module (rather than inline in the listener) so it can be unit tested:
 * the listener itself needs a live store and is exercised only at runtime.
 */
export const getVideoUploadFailedDescription = (
  fileName: string | undefined,
  errorMessage: string | undefined
): string | undefined => {
  const parts = [fileName, errorMessage].filter(Boolean);
  return parts.length > 0 ? parts.join(': ') : undefined;
};
