/**
 * True when the server has answered that this client cannot have the video: it is gone, or it
 * is not theirs to read. Both mean the reference is unusable and the field holding it should
 * let it go.
 *
 * Two statuses, because "deleted" does not always arrive as 404. `_assert_video_read_access`
 * decides on `videos.user_id`, which disappears with the row, so in a multiuser deployment a
 * deleted video is indistinguishable from someone else's and both are refused with 403 — only
 * an admin (and so every single-user deployment, whose default user is one) reaches the read
 * that 404s. Treating 404 alone as gone would strand a deleted video in every workflow field
 * that references it.
 *
 * Everything else is indeterminate and must NOT discard the user's input: a transient network
 * error (`FETCH_ERROR`), an auth failure (401), a timeout or a 5xx says nothing about whether
 * the video exists, and the reset is silent and has no undo.
 *
 * The 403 arm carries a matching obligation on the server, met in `_assert_video_read_access`:
 * a storage error must never be laundered into a 403, or an unreadable database would present
 * as a permission decision and take the user's references down with it. `isImageUnavailableError`
 * is the same predicate for images.
 */
export const isVideoUnavailableError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && (error.status === 404 || error.status === 403);
