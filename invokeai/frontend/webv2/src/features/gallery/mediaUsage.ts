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
export { checkGalleryMediaUsage } from './data/backend';
