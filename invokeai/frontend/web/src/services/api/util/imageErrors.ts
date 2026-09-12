/**
 * True only for a confirmed "this image no longer exists" (HTTP 404).
 *
 * Several components drop the user's reference when the image behind it is gone — a node's
 * image field, a global reference image, a regional guidance reference image. That reset is
 * silent and has no undo, so it may only follow an answer that is both definite and permanent.
 *
 * A 403 is neither, which is why it is excluded even though it is the answer a deleted image
 * *used* to produce for a non-admin. Access is revoked and restored: flip a board to Private
 * and every reference to its images would clear; flip it back and the images are all still
 * there, but the workflows that pointed at them are not. The server draws the distinction
 * instead — `assert_image_read_access` answers 404 for an image positively absent and 403 only
 * for one it is refusing — and that is what makes 404 alone the right test here.
 *
 * Everything else is indeterminate and must NOT discard the user's input: a transient network
 * failure (`FETCH_ERROR`), a timeout, a parse failure, a 401 or a 5xx says nothing about
 * whether the image exists. That became load-bearing when the star/unstar mutations began
 * invalidating the DTOs of names reported in `failed_images`: a name is reported there precisely
 * because a storage failure interrupted its write, so the refetch the invalidation triggers is
 * running against a store that is already unwell and is likelier than usual to answer 500.
 *
 * The server side carries the matching obligation, met in `image_records.get`: a storage error
 * must never be translated into not-found, or an unreadable database would present as a
 * deleted image and take the user's references down with it.
 */
export const isImageMissingError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && error.status === 404;
