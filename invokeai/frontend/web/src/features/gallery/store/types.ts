import type { ImageCategory, ImageDTO } from 'services/api/types';

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = ['control', 'mask', 'user', 'other'];
export const IMAGE_LIMIT = 15;

export type GalleryView = 'images' | 'assets';
export type BoardId = 'none' | (string & Record<never, never>);
export type ComparisonMode = 'slider' | 'side-by-side' | 'hover';
export type ComparisonFit = 'contain' | 'fill';

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
  alwaysShowImageSizeBadge: boolean;
  imageToCompare: ImageDTO | null;
  comparisonMode: ComparisonMode;
  comparisonFit: ComparisonFit;
  isImageViewerOpen: boolean;
  shouldShowArchivedBoards: boolean;
};
