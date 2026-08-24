/**
 * True when the server has answered that this client cannot have the image: it is gone, or it
 * is not theirs to read. Both mean the reference is unusable and the component holding it
 * should let it go.
 *
 * Two statuses, because "deleted" does not always arrive as 404. `assert_image_read_access`
 * decides on `images.user_id`, which disappears with the row, so in a multiuser deployment a
 * deleted image is indistinguishable from someone else's and both are refused with 403 — only
 * an admin (and so every single-user deployment, whose default user is one) reaches the read
 * that 404s. Treating 404 alone as gone would strand a deleted image in every workflow field
 * that references it.
 *
 * Everything else is indeterminate and must NOT discard the user's input. A transient network
 * failure (`FETCH_ERROR`), a timeout, a parse failure or a 5xx says nothing about whether the
 * image exists, and this reset is silent and has no undo. That distinction became load-bearing
 * when the star/unstar mutations began invalidating the DTOs of names the server reported in
 * `failed_images`: a name is reported there precisely because a storage failure interrupted its
 * write, so the refetch the invalidation triggers is running against a store that is already
 * unwell and is likelier than usual to answer 500.
 *
 * The 403 arm carries a matching obligation on the server, met in `assert_image_read_access`:
 * a storage error must never be laundered into a 403, or an unreadable database would present
 * as a permission decision and take the user's references down with it.
 */
export const isImageUnavailableError = (error: unknown): boolean =>
  error instanceof Object && 'status' in error && (error.status === 404 || error.status === 403);
