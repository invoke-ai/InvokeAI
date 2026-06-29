import type { GalleryBoardOrderBy, GalleryOrderDir } from './api';

export type GalleryThumbnailFit = 'square' | 'aspect';

export type GalleryPaginationMode = 'infinite' | 'paginated';

/**
 * User-tunable gallery settings. Persisted as plain entries in the gallery
 * widget's `values` record so they survive workbench autosave/hydration.
 */
export interface GallerySettings {
  boardOrderBy: GalleryBoardOrderBy;
  boardOrderDir: GalleryOrderDir;
  imageDensityPercent: number;
  imageOrderDir: GalleryOrderDir;
  paginationMode: GalleryPaginationMode;
  showArchivedBoards: boolean;
  showDateBoards: boolean;
  showImageDimensions: boolean;
  starredFirst: boolean;
  thumbnailFit: GalleryThumbnailFit;
}

export const DEFAULT_GALLERY_SETTINGS: GallerySettings = {
  boardOrderBy: 'created_at',
  boardOrderDir: 'DESC',
  imageDensityPercent: 50,
  imageOrderDir: 'DESC',
  paginationMode: 'infinite',
  showArchivedBoards: false,
  showDateBoards: false,
  showImageDimensions: false,
  starredFirst: true,
  thumbnailFit: 'square',
};

const isOrderDir = (value: unknown): value is GalleryOrderDir => value === 'ASC' || value === 'DESC';

const isBoardOrderBy = (value: unknown): value is GalleryBoardOrderBy =>
  value === 'created_at' || value === 'board_name';

export const getGallerySettings = (values: Record<string, unknown>): GallerySettings => ({
  boardOrderBy: isBoardOrderBy(values.boardOrderBy) ? values.boardOrderBy : DEFAULT_GALLERY_SETTINGS.boardOrderBy,
  boardOrderDir: isOrderDir(values.boardOrderDir) ? values.boardOrderDir : DEFAULT_GALLERY_SETTINGS.boardOrderDir,
  imageDensityPercent:
    typeof values.imageDensityPercent === 'number' && Number.isFinite(values.imageDensityPercent)
      ? Math.min(100, Math.max(0, values.imageDensityPercent))
      : DEFAULT_GALLERY_SETTINGS.imageDensityPercent,
  imageOrderDir: isOrderDir(values.imageOrderDir) ? values.imageOrderDir : DEFAULT_GALLERY_SETTINGS.imageOrderDir,
  paginationMode: values.paginationMode === 'paginated' ? 'paginated' : DEFAULT_GALLERY_SETTINGS.paginationMode,
  showArchivedBoards:
    typeof values.showArchivedBoards === 'boolean'
      ? values.showArchivedBoards
      : DEFAULT_GALLERY_SETTINGS.showArchivedBoards,
  showDateBoards:
    typeof values.showDateBoards === 'boolean' ? values.showDateBoards : DEFAULT_GALLERY_SETTINGS.showDateBoards,
  showImageDimensions:
    typeof values.showImageDimensions === 'boolean'
      ? values.showImageDimensions
      : DEFAULT_GALLERY_SETTINGS.showImageDimensions,
  starredFirst: typeof values.starredFirst === 'boolean' ? values.starredFirst : DEFAULT_GALLERY_SETTINGS.starredFirst,
  thumbnailFit: values.thumbnailFit === 'aspect' ? 'aspect' : DEFAULT_GALLERY_SETTINGS.thumbnailFit,
});
