import type { GalleryBoard } from './types';

export type GalleryBoardTranslate = (key: string) => string;

/** Resolves the localized display label for synthesized and stored boards. */
export const getGalleryBoardLabel = (board: GalleryBoard, t: GalleryBoardTranslate): string =>
  board.kind === 'uncategorized' ? t('widgets.gallery.uncategorized') : board.name;
