/**
 * Server image storage administration: the subfolder layout and the move/recovery job. Separate from `queries` so
 * the editor's gallery reads never load this admin-only transport.
 */
export {
  getImageMoveStatus,
  getImageStorageLayout,
  IMAGE_STORAGE_LAYOUTS,
  imageMoveStatusQueryOptions,
  imageStorageKeys,
  imageStorageLayoutQueryOptions,
  isImageStorageLayout,
  isTerminalImageMoveJobState,
  startImageMove,
  startImageMoveRecovery,
  updateImageStorageLayout,
} from './data/imageStorage';
export type {
  ImageMoveJob,
  ImageMoveJobState,
  ImageMoveOperation,
  ImageMoveStatus,
  ImageStorageLayout,
} from './data/imageStorage';
