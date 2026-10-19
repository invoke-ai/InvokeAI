import type { GalleryImage, GeneratedImageContract } from './types';

type LocalGalleryImage = GeneratedImageContract & Partial<Pick<GalleryImage, 'boardId' | 'imageCategory' | 'starred'>>;

export const normalizeGalleryImage = (image: LocalGalleryImage, fallbackBoardId = 'none'): GalleryImage => ({
  ...image,
  boardId: image.boardId ?? fallbackBoardId,
  imageCategory: image.imageCategory ?? 'general',
  starred: image.starred ?? false,
});
