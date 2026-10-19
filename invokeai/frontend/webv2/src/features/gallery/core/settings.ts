import type { GalleryItemKind } from './items';
import type { GalleryBoardOrderBy, GalleryOrderDir } from './types';

export type GalleryThumbnailFit = 'square' | 'aspect';

export type GalleryPaginationMode = 'infinite' | 'paginated';

/** Which media kinds the gallery lists: both, or only one. */
export type GalleryMediaFilter = 'all' | 'image' | 'video';

/** Reads only the media filter, for subscribers that run on every Workbench change; unknown values read as both. */
export const getGalleryMediaFilter = (values: Record<string, unknown>): GalleryMediaFilter =>
  values.mediaFilter === 'image' || values.mediaFilter === 'video' ? values.mediaFilter : 'all';

/** The listing's media-kind restriction; undefined lists both kinds. */
export const toGalleryMediaKind = (filter: GalleryMediaFilter): GalleryItemKind | undefined =>
  filter === 'all' ? undefined : filter;

/** The collapsible groups in the board panel, in render order. */
export const GALLERY_BOARD_SECTION_IDS = [
  'project',
  'library',
  'shared',
  'other-users',
  'other-projects',
  'dates',
  'archived',
] as const;

export type GalleryBoardSectionId = (typeof GALLERY_BOARD_SECTION_IDS)[number];

/**
 * Groups that start collapsed, which the panel records as opened rather than collapsed: an admin's view of other
 * users' boards stays out of the way of their own on a busy install.
 */
const GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT: readonly GalleryBoardSectionId[] = ['other-users'];

/** Height in the stacked layout, width in the wide one — stored separately. */
export const GALLERY_BOARD_PANEL_MIN_HEIGHT_PX = 120;
export const GALLERY_BOARD_PANEL_MAX_HEIGHT_PX = 600;
export const GALLERY_MIN_GRID_HEIGHT_PX = 128;
export const GALLERY_BOARD_PANEL_MIN_WIDTH_PX = 180;
export const GALLERY_BOARD_PANEL_MAX_WIDTH_PX = 420;

/** The auto-add choice that sends results to whichever board the gallery has selected. */
export const GALLERY_AUTO_ADD_FOLLOW = 'follow';

/** User-tunable gallery settings, persisted in the widget's `values` record. */
export interface GallerySettings {
  /**
   * Where results without a board of their own go: `follow` (the selected board), `none` (Uncategorized), or a
   * board id. Workflow nodes set to Auto use it too.
   */
  autoAddBoardId: string;
  boardOrderBy: GalleryBoardOrderBy;
  boardOrderDir: GalleryOrderDir;
  /** Board panel disclosure controlled by the region-independent widget header. */
  boardPanelCollapsed: boolean;
  boardPanelHeightPx: number;
  boardPanelWidthPx: number;
  collapsedBoardSections: GalleryBoardSectionId[];
  /** The groups that start collapsed which have been opened. */
  expandedBoardSections: GalleryBoardSectionId[];
  imageDensityPercent: number;
  imageOrderDir: GalleryOrderDir;
  mediaFilter: GalleryMediaFilter;
  /** Opt-in: a middle click on a tile opens the media in a new browser tab instead of the browser default. */
  middleClickOpensInNewTab: boolean;
  paginationMode: GalleryPaginationMode;
  showArchivedBoards: boolean;
  showDateBoards: boolean;
  showImageDimensions: boolean;
  /** Hide readable, project-labelled boards belonging to other projects. */
  showOtherProjectBoards: boolean;
  showPendingItems: boolean;
  progressSectionCollapsed: boolean;
  /** Disclosure of the starred strip at the top of the grid. */
  starredSectionCollapsed: boolean;
  thumbnailFit: GalleryThumbnailFit;
}

export const DEFAULT_GALLERY_SETTINGS: GallerySettings = {
  autoAddBoardId: GALLERY_AUTO_ADD_FOLLOW,
  boardOrderBy: 'created_at',
  boardOrderDir: 'DESC',
  boardPanelCollapsed: false,
  boardPanelHeightPx: 280,
  boardPanelWidthPx: 240,
  collapsedBoardSections: [],
  expandedBoardSections: [],
  imageDensityPercent: 50,
  imageOrderDir: 'DESC',
  mediaFilter: 'all',
  middleClickOpensInNewTab: false,
  paginationMode: 'infinite',
  showArchivedBoards: false,
  showDateBoards: false,
  showImageDimensions: false,
  showOtherProjectBoards: false,
  showPendingItems: true,
  progressSectionCollapsed: false,
  starredSectionCollapsed: false,
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
const getBoardSections = (value: unknown, candidates: readonly GalleryBoardSectionId[]): GalleryBoardSectionId[] =>
  Array.isArray(value)
    ? candidates.filter((sectionId) => value.some((entry) => isBoardSectionId(entry) && entry === sectionId))
    : [];

export const isGalleryBoardSectionOpen = (settings: GallerySettings, sectionId: GalleryBoardSectionId): boolean =>
  GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT.includes(sectionId)
    ? settings.expandedBoardSections.includes(sectionId)
    : !settings.collapsedBoardSections.includes(sectionId);

/** The settings change that opens or closes a group. */
export const toggleGalleryBoardSection = (
  settings: GallerySettings,
  sectionId: GalleryBoardSectionId,
  isOpen: boolean
): Pick<GallerySettings, 'collapsedBoardSections'> | Pick<GallerySettings, 'expandedBoardSections'> => {
  const toggle = (sections: GalleryBoardSectionId[], include: boolean) =>
    include
      ? [...sections.filter((entry) => entry !== sectionId), sectionId]
      : sections.filter((entry) => entry !== sectionId);
  return GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT.includes(sectionId)
    ? { expandedBoardSections: toggle(settings.expandedBoardSections, isOpen) }
    : { collapsedBoardSections: toggle(settings.collapsedBoardSections, !isOpen) };
};

export const getGallerySettings = (values: Record<string, unknown>): GallerySettings => {
  const paginationMode: GalleryPaginationMode =
    values.paginationMode === 'paginated' ? 'paginated' : DEFAULT_GALLERY_SETTINGS.paginationMode;

  return {
    autoAddBoardId:
      typeof values.autoAddBoardId === 'string' && values.autoAddBoardId !== ''
        ? values.autoAddBoardId
        : DEFAULT_GALLERY_SETTINGS.autoAddBoardId,
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
    collapsedBoardSections: getBoardSections(
      values.collapsedBoardSections,
      GALLERY_BOARD_SECTION_IDS.filter((sectionId) => !GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT.includes(sectionId))
    ),
    expandedBoardSections: getBoardSections(values.expandedBoardSections, GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT),
    imageDensityPercent:
      typeof values.imageDensityPercent === 'number' && Number.isFinite(values.imageDensityPercent)
        ? Math.min(100, Math.max(0, values.imageDensityPercent))
        : DEFAULT_GALLERY_SETTINGS.imageDensityPercent,
    imageOrderDir: isOrderDir(values.imageOrderDir) ? values.imageOrderDir : DEFAULT_GALLERY_SETTINGS.imageOrderDir,
    mediaFilter: getGalleryMediaFilter(values),
    middleClickOpensInNewTab: values.middleClickOpensInNewTab === true,
    paginationMode,
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
      typeof values.showPendingItems === 'boolean'
        ? values.showPendingItems
        : DEFAULT_GALLERY_SETTINGS.showPendingItems,
    progressSectionCollapsed: values.progressSectionCollapsed === true,
    starredSectionCollapsed:
      typeof values.starredSectionCollapsed === 'boolean'
        ? values.starredSectionCollapsed
        : DEFAULT_GALLERY_SETTINGS.starredSectionCollapsed,
    thumbnailFit: values.thumbnailFit === 'aspect' ? 'aspect' : DEFAULT_GALLERY_SETTINGS.thumbnailFit,
  };
};
