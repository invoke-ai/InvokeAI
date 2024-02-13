import type { ImageCategory, ImageDTO } from 'services/api/types';

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = ['control', 'mask', 'user', 'other'];
export const INITIAL_IMAGE_LIMIT = 100;
export const IMAGE_LIMIT = 20;

export type GalleryView = 'images' | 'assets';
export type BoardId = 'none' | (string & Record<never, never>);

export type GalleryState = {
  selection: ImageDTO[];
  shouldAutoSwitch: boolean;
  autoAssignBoardOnClick: boolean;
  autoAddBoardId: BoardId;
  galleryImageMinimumWidth: number;
  selectedBoardId: BoardId;
  galleryView: GalleryView;
  boardSearchText: string;
  offset: number;
  limit: number;
  uploadedImages: string[];
};
