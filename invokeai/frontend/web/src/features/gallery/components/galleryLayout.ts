import type { GalleryLayoutMode } from 'features/gallery/store/types';

export const getEffectiveGalleryLayout = (
  galleryLayoutMode: GalleryLayoutMode,
  shouldUsePagedGalleryView: boolean
): GalleryLayoutMode => {
  if (shouldUsePagedGalleryView) {
    return 'grid';
  }
  return galleryLayoutMode;
};
