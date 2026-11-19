import type { GalleryItemKind } from './items';
import type { GalleryBoardOrderBy, GalleryOrderDir } from './types';

export type GalleryThumbnailFit = 'square' | 'aspect';

export type GalleryPaginationMode = 'infinite' | 'paginated';

export type GalleryMediaFilter = 'all' | 'image' | 'video';

/** For subscribers that run on every Workbench change; unknown values read as both. */
export const getGalleryMediaFilter = (values: Record<string, unknown>): GalleryMediaFilter =>
  values.mediaFilter === 'image' || values.mediaFilter === 'video' ? values.mediaFilter : 'all';

export const toGalleryMediaKind = (filter: GalleryMediaFilter): GalleryItemKind | undefined =>
  filter === 'all' ? undefined : filter;

/** Applies only while a search is active; clearing the search returns it to `board`. */
export type GallerySearchScope = 'board' | 'all';

/** The newest are kept. */
export const GALLERY_COLLAPSED_PROJECT_GROUPS_LIMIT = 200;

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

/** Recorded as opened rather than collapsed: other users' boards stay out of an admin's way. */
const GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT: readonly GalleryBoardSectionId[] = ['other-users'];

/** Height in the stacked layout, width in the wide one, stored separately. */
export const GALLERY_BOARD_PANEL_MIN_HEIGHT_PX = 120;
export const GALLERY_BOARD_PANEL_MAX_HEIGHT_PX = 600;
export const GALLERY_MIN_GRID_HEIGHT_PX = 128;
export const GALLERY_BOARD_PANEL_MIN_WIDTH_PX = 180;
export const GALLERY_BOARD_PANEL_MAX_WIDTH_PX = 420;

/** Earlier builds' stored "follow" reads as the default destination; following is now `followSelectedBoard`. */
const LEGACY_AUTO_ADD_FOLLOW = 'follow';

export const isLegacyFollowAutoAdd = (values: Readonly<Record<string, unknown>>): boolean =>
  values.autoAddBoardId === LEGACY_AUTO_ADD_FOLLOW;

export interface GallerySettings {
  /** `none` is Uncategorized; null is the project's inbox. Workflow nodes set to Auto use it too. */
  autoAddBoardId: string | null;
  boardOrderBy: GalleryBoardOrderBy;
  boardOrderDir: GalleryOrderDir;
  boardPanelCollapsed: boolean;
  boardPanelHeightPx: number;
  boardPanelWidthPx: number;
  collapsedBoardSections: GalleryBoardSectionId[];
  /** Oldest first. */
  collapsedProjectGroupIds: string[];
  expandedBoardSections: GalleryBoardSectionId[];
  /** Replaces `autoAddBoardId` while on. */
  followSelectedBoard: boolean;
  imageDensityPercent: number;
  imageOrderDir: GalleryOrderDir;
  mediaFilter: GalleryMediaFilter;
  middleClickOpensInNewTab: boolean;
  paginationMode: GalleryPaginationMode;
  showArchivedBoards: boolean;
  showDateBoards: boolean;
  showImageDimensions: boolean;
  /** Hide readable, project-labelled boards belonging to other projects. */
  showOtherProjectBoards: boolean;
  showPendingItems: boolean;
  progressSectionCollapsed: boolean;
  searchScope: GallerySearchScope;
  starredSectionCollapsed: boolean;
  thumbnailFit: GalleryThumbnailFit;
}

export const DEFAULT_GALLERY_SETTINGS: GallerySettings = {
  autoAddBoardId: null,
  boardOrderBy: 'created_at',
  boardOrderDir: 'DESC',
  boardPanelCollapsed: false,
  boardPanelHeightPx: 280,
  boardPanelWidthPx: 240,
  collapsedBoardSections: [],
  collapsedProjectGroupIds: [],
  expandedBoardSections: [],
  followSelectedBoard: false,
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
  searchScope: 'board',
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

/** A stale id would close a section with no way back. */
const getBoardSections = (value: unknown, candidates: readonly GalleryBoardSectionId[]): GalleryBoardSectionId[] =>
  Array.isArray(value)
    ? candidates.filter((sectionId) => value.some((entry) => isBoardSectionId(entry) && entry === sectionId))
    : [];

/** One pass from the end stops at the limit, so a huge imported list costs no more than the ids kept. */
const getCollapsedProjectGroupIds = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const kept = new Set<string>();

  for (let index = value.length - 1; index >= 0 && kept.size < GALLERY_COLLAPSED_PROJECT_GROUPS_LIMIT; index--) {
    const entry: unknown = value[index];

    if (typeof entry === 'string') {
      kept.add(entry);
    }
  }

  return [...kept].reverse();
};

export const toggleGalleryProjectGroup = (
  settings: GallerySettings,
  projectId: string,
  isOpen: boolean
): Pick<GallerySettings, 'collapsedProjectGroupIds'> => {
  const others = settings.collapsedProjectGroupIds.filter((id) => id !== projectId);

  return {
    collapsedProjectGroupIds: isOpen ? others : [...others, projectId].slice(-GALLERY_COLLAPSED_PROJECT_GROUPS_LIMIT),
  };
};

export const isGalleryBoardSectionOpen = (settings: GallerySettings, sectionId: GalleryBoardSectionId): boolean =>
  GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT.includes(sectionId)
    ? settings.expandedBoardSections.includes(sectionId)
    : !settings.collapsedBoardSections.includes(sectionId);

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
      typeof values.autoAddBoardId === 'string' && values.autoAddBoardId !== '' && !isLegacyFollowAutoAdd(values)
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
    collapsedProjectGroupIds: getCollapsedProjectGroupIds(values.collapsedProjectGroupIds),
    expandedBoardSections: getBoardSections(values.expandedBoardSections, GALLERY_BOARD_SECTIONS_COLLAPSED_BY_DEFAULT),
    followSelectedBoard: values.followSelectedBoard === true,
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
    searchScope: values.searchScope === 'all' ? 'all' : DEFAULT_GALLERY_SETTINGS.searchScope,
    starredSectionCollapsed:
      typeof values.starredSectionCollapsed === 'boolean'
        ? values.starredSectionCollapsed
        : DEFAULT_GALLERY_SETTINGS.starredSectionCollapsed,
    thumbnailFit: values.thumbnailFit === 'aspect' ? 'aspect' : DEFAULT_GALLERY_SETTINGS.thumbnailFit,
  };
};
