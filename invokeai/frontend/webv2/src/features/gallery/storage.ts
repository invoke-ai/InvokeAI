/**
 * Server image storage administration: the subfolder layout and the move/recovery job. Separate from `queries` so
 * the editor's gallery reads never load this admin-only transport.
 */
export {
  IMAGE_STORAGE_LAYOUTS,
  imageMoveStatusQueryOptions,
  imageStorageLayoutQueryOptions,
  isImageStorageLayout,
  startImageMove,
  startImageMoveRecovery,
  updateImageStorageLayout,
} from './data/imageStorage';
export type { ImageMoveStatus, ImageStorageLayout } from './data/imageStorage';
