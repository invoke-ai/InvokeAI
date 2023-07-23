import { ImageCategory } from 'services/api/types';

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = [
  'control',
  'mask',
  'user',
  'other',
];
export const INITIAL_IMAGE_LIMIT = 100;
export const IMAGE_LIMIT = 20;

export type GalleryView = 'images' | 'assets';
// export type BoardId = 'no_board' | (string & Record<never, never>);
export type BoardId = string | undefined;

export type GalleryState = {
  selection: string[];
  shouldAutoSwitch: boolean;
  autoAddBoardId: string | undefined;
  galleryImageMinimumWidth: number;
  selectedBoardId: BoardId;
  galleryView: GalleryView;
  batchImageNames: string[];
  isBatchEnabled: boolean;
};
