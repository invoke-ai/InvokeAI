import type { GalleryBoardOrderBy, GalleryOrderDir } from './types';

export type GalleryThumbnailFit = 'square' | 'aspect';

export type GalleryPaginationMode = 'infinite' | 'paginated';

/** The collapsible groups in the board panel, in render order. */
export const GALLERY_BOARD_SECTION_IDS = ['boards', 'dates', 'archived'] as const;

export type GalleryBoardSectionId = (typeof GALLERY_BOARD_SECTION_IDS)[number];

/** Height in the stacked layout, width in the wide one — stored separately. */
export const GALLERY_BOARD_PANEL_MIN_HEIGHT_PX = 120;
export const GALLERY_BOARD_PANEL_MAX_HEIGHT_PX = 600;
export const GALLERY_MIN_GRID_HEIGHT_PX = 128;
export const GALLERY_BOARD_PANEL_MIN_WIDTH_PX = 180;
export const GALLERY_BOARD_PANEL_MAX_WIDTH_PX = 420;

/**
 * User-tunable gallery settings. Persisted as plain entries in the gallery
 * widget's `values` record so they survive workbench autosave/hydration.
 */
export interface GallerySettings {
  boardOrderBy: GalleryBoardOrderBy;
  boardOrderDir: GalleryOrderDir;
  /** Board panel disclosure controlled by the region-independent widget header. */
  boardPanelCollapsed: boolean;
  boardPanelHeightPx: number;
  boardPanelWidthPx: number;
  collapsedBoardSections: GalleryBoardSectionId[];
  imageDensityPercent: number;
  imageOrderDir: GalleryOrderDir;
  paginationMode: GalleryPaginationMode;
  showArchivedBoards: boolean;
  showDateBoards: boolean;
  showImageDimensions: boolean;
  /**
   * Boards owned by another project. They are readable and already labelled
   * with a project badge; this hides them for anyone who wants the panel to
   * be about the project they have open.
   */
  showOtherProjectBoards: boolean;
  showPendingItems: boolean;
  starredFirst: boolean;
  thumbnailFit: GalleryThumbnailFit;
}

export const DEFAULT_GALLERY_SETTINGS: GallerySettings = {
  boardOrderBy: 'created_at',
  boardOrderDir: 'DESC',
  boardPanelCollapsed: false,
  boardPanelHeightPx: 280,
  boardPanelWidthPx: 240,
  collapsedBoardSections: [],
  imageDensityPercent: 50,
  imageOrderDir: 'DESC',
  paginationMode: 'infinite',
  showArchivedBoards: false,
  showDateBoards: false,
  showImageDimensions: false,
  showOtherProjectBoards: true,
  showPendingItems: true,
  starredFirst: true,
  thumbnailFit: 'square',
};

const isOrderDir = (value: unknown): value is GalleryOrderDir => value === 'ASC' || value === 'DESC';

const isBoardOrderBy = (value: unknown): value is GalleryBoardOrderBy =>
  value === 'created_at' || value === 'board_name';

const isBoardSectionId = (value: unknown): value is GalleryBoardSectionId =>
  GALLERY_BOARD_SECTION_IDS.includes(value as GalleryBoardSectionId);

const getBoundedSize = (
  value: unknown,
  { fallback, max, min }: { fallback: number; max: number; min: number }
): number =>
  typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, Math.round(value))) : fallback;

/** Filtered to known ids: a stale one would close a section with no way back. */
const getCollapsedBoardSections = (value: unknown): GalleryBoardSectionId[] =>
  Array.isArray(value)
    ? GALLERY_BOARD_SECTION_IDS.filter((sectionId) =>
        value.some((entry) => isBoardSectionId(entry) && entry === sectionId)
      )
    : DEFAULT_GALLERY_SETTINGS.collapsedBoardSections;

export const getGallerySettings = (values: Record<string, unknown>): GallerySettings => ({
  boardOrderBy: isBoardOrderBy(values.boardOrderBy) ? values.boardOrderBy : DEFAULT_GALLERY_SETTINGS.boardOrderBy,
  boardOrderDir: isOrderDir(values.boardOrderDir) ? values.boardOrderDir : DEFAULT_GALLERY_SETTINGS.boardOrderDir,
  boardPanelCollapsed:
    typeof values.boardPanelCollapsed === 'boolean'
      ? values.boardPanelCollapsed
      : DEFAULT_GALLERY_SETTINGS.boardPanelCollapsed,
  boardPanelHeightPx: getBoundedSize(values.boardPanelHeightPx, {
    fallback: DEFAULT_GALLERY_SETTINGS.boardPanelHeightPx,
    max: GALLERY_BOARD_PANEL_MAX_HEIGHT_PX,
    min: GALLERY_BOARD_PANEL_MIN_HEIGHT_PX,
  }),
  boardPanelWidthPx: getBoundedSize(values.boardPanelWidthPx, {
    fallback: DEFAULT_GALLERY_SETTINGS.boardPanelWidthPx,
    max: GALLERY_BOARD_PANEL_MAX_WIDTH_PX,
    min: GALLERY_BOARD_PANEL_MIN_WIDTH_PX,
  }),
  collapsedBoardSections: getCollapsedBoardSections(values.collapsedBoardSections),
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
  showOtherProjectBoards:
    typeof values.showOtherProjectBoards === 'boolean'
      ? values.showOtherProjectBoards
      : DEFAULT_GALLERY_SETTINGS.showOtherProjectBoards,
  showPendingItems:
    typeof values.showPendingItems === 'boolean' ? values.showPendingItems : DEFAULT_GALLERY_SETTINGS.showPendingItems,
  // Starred items have a dedicated section, so the backing window must always
  // fetch them first to keep that section complete under infinite pagination.
  starredFirst: true,
  thumbnailFit: values.thumbnailFit === 'aspect' ? 'aspect' : DEFAULT_GALLERY_SETTINGS.thumbnailFit,
});
