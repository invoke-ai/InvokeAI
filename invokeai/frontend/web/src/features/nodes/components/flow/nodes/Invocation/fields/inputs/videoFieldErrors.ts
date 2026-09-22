/**
 * True only for a confirmed "video does not exist" (HTTP 404) query error.
 *
 * VideoFieldInputComponent clears its field value when the referenced video is gone, but
 * a transient network error (`FETCH_ERROR`), auth failure (401/403), or server error (5xx)
 * must not silently discard the user's input — only a 404 proves the video was deleted.
 */
export const isVideoMissingError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && error.status === 404;
