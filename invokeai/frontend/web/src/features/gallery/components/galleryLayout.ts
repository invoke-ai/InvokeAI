import type { GalleryLayoutMode } from 'features/gallery/store/types';

export type GalleryContentView = 'paged' | GalleryLayoutMode;

export const getEffectiveGalleryLayout = (
  galleryLayoutMode: GalleryLayoutMode,
  shouldUsePagedGalleryView: boolean
): GalleryLayoutMode => {
  if (shouldUsePagedGalleryView) {
    return 'grid';
  }
  return galleryLayoutMode;
};

export const getGalleryContentView = (
  galleryLayoutMode: GalleryLayoutMode,
  shouldUsePagedGalleryView: boolean
): GalleryContentView => {
  if (shouldUsePagedGalleryView) {
    return 'paged';
  }
  return galleryLayoutMode;
};
