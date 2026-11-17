/** Where media about to be deleted is still used: the server check and the report types. */
export {
  isGalleryMediaUnused,
  type GalleryMediaUsage,
  type GalleryMediaUsageCheck,
  type GalleryMediaUsageDocument,
  type GalleryMediaUsageGroup,
  type GalleryMediaUsageReport,
  type GalleryMediaUsageRequest,
  type GalleryMediaUsageSettledCheck,
  type GalleryMediaUsageSurface,
  type GalleryMediaUsageSurfaceRow,
} from './core/mediaUsage';
export { GALLERY_REQUEST_MAX_NAMES } from './data/backend';
export { checkGalleryMediaUsage } from './data/mediaUsage';
