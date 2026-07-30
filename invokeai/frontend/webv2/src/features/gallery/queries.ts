export {
  canonicalizeGalleryItemsFilter,
  flattenGalleryItemsData,
  GALLERY_MAX_INFINITE_PAGES,
  GALLERY_MAX_ROWS,
  GALLERY_PAGE_SIZE,
  galleryBoardsOptions,
  galleryItemNamesOptions,
  galleryItemsInfiniteOptions,
  galleryKeys,
} from './data/queries';
export type {
  CanonicalGalleryItemsFilter,
  GalleryBoardsQuery,
  GalleryItemsFilter,
  GalleryItemsListQueryKey,
  GalleryItemsWindow,
} from './data/queries';

/**
 * TODO(Task 5/7): Remove these legacy image-query facade exports after Gallery
 * state/projection and Preview consume the mixed item query API.
 */
export {
  canonicalizeGalleryImagesFilter,
  flattenGalleryImagesData,
  galleryImagesInfiniteOptions,
} from './data/queries';
export type {
  CanonicalGalleryImagesFilter,
  GalleryImagesFilter,
  GalleryImagesListQueryKey,
  GalleryImagesWindow,
} from './data/queries';
export { invalidateGallery, invalidateGalleryItems, patchGalleryItemCaches } from './data/queryCache';
export type { GalleryItemCachePatch } from './data/queryCache';

/**
 * TODO(Task 5/7): Remove these legacy image-cache facade exports after image
 * actions and remaining image-only consumers use confirmed mixed item results.
 */
export { invalidateGalleryImages, patchGalleryImageCaches } from './data/queryCache';
export type { GalleryImageCachePatch } from './data/queryCache';
