/**
 * True only for a confirmed "this video no longer exists" (HTTP 404).
 *
 * `VideoFieldInputComponent` drops the user's reference when the video behind it is gone. That
 * reset is silent and has no undo, so it may only follow an answer that is both definite and
 * permanent — which a 403 is not. Access is revoked and restored: flip a board to Private and
 * every field referencing its videos would clear; flip it back and the videos are all still
 * there, but the workflows that pointed at them are not.
 *
 * `_assert_video_read_access` draws that distinction server-side, answering 404 for a video
 * positively absent and 403 only for one it is refusing, and `video_records.get` no longer
 * translates storage errors into not-found — without which an unreadable database would present
 * as a deleted video. `isImageMissingError` is the same predicate for images.
 */
export const isVideoMissingError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && error.status === 404;
