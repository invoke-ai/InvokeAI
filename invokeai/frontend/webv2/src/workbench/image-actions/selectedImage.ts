import type { GalleryImage } from '@workbench/gallery/api';
import type { GeneratedImageContract, Project } from '@workbench/types';

import { getProjectWidgetValues } from '@workbench/widgetState';

const isGeneratedImage = (value: unknown): value is GeneratedImageContract =>
  Boolean(value) && typeof value === 'object' && typeof (value as GeneratedImageContract).imageName === 'string';

const toGalleryImage = (
  image: GeneratedImageContract & Partial<GalleryImage>,
  galleryValues: Record<string, unknown>
): GalleryImage => ({
  ...image,
  boardId:
    image.boardId ?? (typeof galleryValues.selectedBoardId === 'string' ? galleryValues.selectedBoardId : 'none'),
  imageCategory: image.imageCategory ?? 'general',
  starred: image.starred ?? false,
});

export const getSelectedGalleryImageFromValues = (galleryValues: Record<string, unknown>): GalleryImage | null => {
  if (isGeneratedImage(galleryValues.selectedImage)) {
    return toGalleryImage(galleryValues.selectedImage, galleryValues);
  }

  const selectedImageName =
    typeof galleryValues.selectedImageName === 'string' ? galleryValues.selectedImageName : null;
  const recentImages = Array.isArray(galleryValues.recentImages) ? galleryValues.recentImages : [];
  const recentImage = recentImages.find(
    (image): image is GeneratedImageContract => isGeneratedImage(image) && image.imageName === selectedImageName
  );

  return recentImage ? toGalleryImage(recentImage, galleryValues) : null;
};

export const getSelectedGalleryImage = (project: Project): GalleryImage | null =>
  getSelectedGalleryImageFromValues(getProjectWidgetValues(project, 'gallery'));
