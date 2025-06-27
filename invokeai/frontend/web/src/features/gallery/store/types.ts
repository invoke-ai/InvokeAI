import type { BoardRecordOrderBy, ImageCategory } from 'services/api/types';

export const IMAGE_CATEGORIES: ImageCategory[] = ['general'];
export const ASSETS_CATEGORIES: ImageCategory[] = ['control', 'mask', 'user', 'other'];

export type GalleryView = 'images' | 'assets';
export type BoardId = 'none' | (string & Record<never, never>);
export type ComparisonMode = 'slider' | 'side-by-side' | 'hover';
export type ComparisonFit = 'contain' | 'fill';
export type OrderDir = 'ASC' | 'DESC';

export type GalleryState = {
  selection: string[];
  shouldAutoSwitch: boolean;
  autoAssignBoardOnClick: boolean;
  autoAddBoardId: BoardId;
  galleryImageMinimumWidth: number;
  selectedBoardId: BoardId;
  galleryView: GalleryView;
  boardSearchText: string;
  starredFirst: boolean;
  orderDir: OrderDir;
  searchTerm: string;
  alwaysShowImageSizeBadge: boolean;
  imageToCompare: string | null;
  comparisonMode: ComparisonMode;
  comparisonFit: ComparisonFit;
  shouldShowArchivedBoards: boolean;
  boardsListOrderBy: BoardRecordOrderBy;
  boardsListOrderDir: OrderDir;
};
