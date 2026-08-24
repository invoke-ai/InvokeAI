/**
 * True only for a confirmed "image does not exist" (HTTP 404) query error.
 *
 * Several components clear the user's input when the image they reference is gone — a node's
 * image field, a global reference image, a regional guidance reference image. A transient
 * network error (`FETCH_ERROR`), an auth failure (401/403), or a server error (5xx) proves
 * nothing about whether the image still exists, and discarding the input over one loses work
 * the user cannot recover: the field is cleared with no toast and no undo.
 *
 * That distinction became load-bearing when the star/unstar mutations began invalidating the
 * DTOs of names the server reported in `failed_images`. Those names are reported precisely
 * because a storage failure interrupted the write, so the refetch that the invalidation triggers
 * is running against a store that is already unwell and is likelier than usual to answer 500.
 *
 * `isVideoMissingError` is the same predicate for videos, and exists for the same reason.
 */
export const isImageMissingError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && error.status === 404;
