import type { GalleryBoard, GalleryImage, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';
import type { ResolvedValues } from '@platform/core/resolvedValues';
import type { IconRadioMenuProps } from '@platform/ui/Menu';
import type { TFunction } from 'i18next';

import {
  ALL_READABLE_BOARDS_ID,
  galleryImageItemToGalleryImage,
  isDateBoardId,
  parseGalleryItemKey,
  toGalleryItemKey,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryItemKind,
} from '@features/gallery/core/items';
import {
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryItemFromValues,
} from '@features/gallery/core/selection';
import {
  gallerySemanticReferenceKey,
  parseGallerySemanticReference,
  type GallerySemanticReference,
} from '@features/gallery/core/semanticImageQuery';
import {
  getGalleryMediaFilter,
  getGallerySettings,
  toGalleryMediaKind,
  type GalleryMediaFilter,
  type GallerySearchScope,
  type GallerySettings,
} from '@features/gallery/core/settings';
import { FilmIcon, ImageIcon, LayersIcon } from 'lucide-react';

const GALLERY_MEDIA_FILTER_OPTIONS = [
  { icon: LayersIcon, labelKey: 'widgets.gallery.mediaFilterAll', value: 'all' },
  { icon: ImageIcon, labelKey: 'widgets.gallery.mediaFilterImages', value: 'image' },
  { icon: FilmIcon, labelKey: 'widgets.gallery.mediaFilterVideos', value: 'video' },
] as const;

/** Lives in this boot-loaded view module, via `contracts`, so lazy widgets add no module. */
export const getGalleryMediaFilterMenuProps = (
  t: TFunction,
  value: GalleryMediaFilter
): Pick<
  IconRadioMenuProps<GalleryMediaFilter>,
  'isActive' | 'label' | 'options' | 'tooltip' | 'triggerAttributes' | 'triggerLabel'
> => {
  const options = GALLERY_MEDIA_FILTER_OPTIONS.map((option) => ({
    icon: option.icon,
    label: t(option.labelKey),
    value: option.value,
  }));
  const currentLabel = options.find((option) => option.value === value)?.label ?? options[0].label;

  return {
    isActive: value !== 'all',
    label: t('widgets.gallery.mediaFilter'),
    options,
    tooltip: t('widgets.gallery.mediaFilterTooltip', { option: currentLabel }),
    triggerAttributes: { 'data-gallery-media-filter-menu': '', 'data-media-filter': value },
    triggerLabel: t('widgets.gallery.mediaFilterTrigger', { option: currentLabel }),
  };
};

/** getGalleryBoardLabel localizes the empty name from kind. */
const UNCATEGORIZED_BOARD: GalleryBoard = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'none',
  imageCount: 0,
  isInbox: false,
  kind: 'uncategorized',
  name: '',
  projectId: null,
  videoCount: 0,
};

export interface GalleryStateView {
  /** A nonzero infinite anchor blocks scrolling to earlier rows; the surface must explain it and offer the top. */
  anchoredWindowPage: number;
  boards: GalleryBoard[];
  compareImageKey: GalleryItemKey | null;
  galleryView: GalleryView;
  isComparisonActive: boolean;
  /** The comparison item is the visible selection itself, waiting for a second item. */
  isCompareArmed: boolean;
  isSearchActive: boolean;
  items: GalleryItem[];
  /** `ALL_READABLE_BOARDS_ID` while a search looks in every board; grid selections are stamped with it. */
  listingBoardId: string;
  page: number;
  projectBoardId: string | null;
  /** Null unless the selection's stamp names the shown listing; reveals follow it across pages. */
  revealTargetPage: number | null;
  searchScope: GallerySearchScope;
  searchTerm: string;
  /** A search of every board leaves it as it is. */
  selectedBoardId: string;
  /** Retained while its sparse page is not loaded. */
  primarySelectedItemKey: GalleryItemKey | null;
  selectedItemKey: GalleryItemKey | null;
  selectedItemKeys: GalleryItemKey[];
  /** Members were starred when selected; a selection carried in from another listing says nothing about star flags. */
  selectionStarredOnly: boolean;
  semanticImageQuery: GallerySemanticReference | null;
  /** Null in metadata mode. */
  semanticSearchText: string | null;
  settings: GallerySettings;
  starredOnly: boolean;
}

export const getGalleryView = (values: Record<string, unknown>): GalleryView =>
  values.galleryView === 'assets' ? 'assets' : 'images';

export const getGallerySearchTerm = (values: Record<string, unknown>): string =>
  typeof values.searchTerm === 'string' ? values.searchTerm : '';

export const getGalleryStarredOnly = (values: Record<string, unknown>): boolean => values.starredOnly === true;

export const getGallerySemanticImageQuery = (values: Record<string, unknown>): GallerySemanticReference | null =>
  parseGallerySemanticReference(values.semanticImageQuery);

export const getGallerySemanticSearchText = (values: Record<string, unknown>): string | null =>
  typeof values.semanticSearchText === 'string' ? values.semanticSearchText : null;

export const isGallerySearchActive = (values: Record<string, unknown>): boolean =>
  getGallerySearchTerm(values).trim() !== '' ||
  getGallerySemanticImageQuery(values) !== null ||
  (getGallerySemanticSearchText(values) ?? '').trim() !== '';

/** With no active search the scope returns to 'board'. Every write of gallery values goes through it. */
export const settleGallerySearchScope = (values: Record<string, unknown>): Record<string, unknown> =>
  values.searchScope === 'all' && !isGallerySearchActive(values) ? { ...values, searchScope: 'board' } : values;

/** Board, view, search, scope and filters agree; order and page aside. */
const isSameGalleryListing = (left: Record<string, unknown>, right: Record<string, unknown>): boolean =>
  getGalleryRawSelectedBoardId(left) === getGalleryRawSelectedBoardId(right) &&
  getGalleryView(left) === getGalleryView(right) &&
  getGallerySearchTerm(left) === getGallerySearchTerm(right) &&
  getGalleryStarredOnly(left) === getGalleryStarredOnly(right) &&
  getGalleryMediaFilter(left) === getGalleryMediaFilter(right) &&
  getGallerySearchScope(left) === getGallerySearchScope(right) &&
  gallerySemanticReferenceKey(getGallerySemanticImageQuery(left)) ===
    gallerySemanticReferenceKey(getGallerySemanticImageQuery(right));

const getPrimaryGalleryItemKey = (values: Record<string, unknown>): GalleryItemKey | null => {
  if (typeof values.selectedImageName === 'string') {
    return toGalleryItemKey(parseGalleryItemKey(values.selectedImageName));
  }

  const selectedItem = getSelectedGalleryItemFromValues(values);

  return selectedItem ? toGalleryItemKey(selectedItem) : null;
};

/** A selection landing on the comparison image would compare it with itself, so the comparison ends instead. */
const movesPrimaryOntoComparison = (previous: Record<string, unknown>, next: Record<string, unknown>): boolean => {
  if (!next.compareImage || next.compareImage !== previous.compareImage) {
    return false;
  }

  const compareItem = getGalleryCompareItem(next);
  const primaryKey = getPrimaryGalleryItemKey(next);

  return (
    compareItem !== null &&
    primaryKey === toGalleryItemKey(compareItem) &&
    primaryKey !== getPrimaryGalleryItemKey(previous)
  );
};

/**
 * Every write settles the search scope, ends a comparison its selection lands on, and drops a selection left in place
 * when the listing changed: Delete must never reach items another board or filter hides. A write that sets the
 * selection itself keeps it.
 */
export const settleGalleryValues = (
  previous: Record<string, unknown>,
  next: Record<string, unknown>
): Record<string, unknown> => {
  const scoped = settleGallerySearchScope(next);
  const settled = movesPrimaryOntoComparison(previous, scoped) ? { ...scoped, compareImage: null } : scoped;
  const setsSelection =
    settled.selectedImageNames !== previous.selectedImageNames ||
    settled.selectedImageQuery !== previous.selectedImageQuery;

  if (
    setsSelection ||
    getPersistedSelectedGalleryItemKeys(settled).length === 0 ||
    isSameGalleryListing(previous, settled)
  ) {
    return settled;
  }

  return { ...settled, selectedImageNames: [] };
};

export const getGallerySearchScope = (values: Record<string, unknown>): GallerySearchScope =>
  values.searchScope === 'all' && isGallerySearchActive(values) ? 'all' : 'board';

export const getGalleryRawSelectedBoardId = (values: Record<string, unknown>): string | null =>
  typeof values.selectedBoardId === 'string' ? values.selectedBoardId : null;

/** Stamping Uncategorized would point Preview at a listing without it. */
export const getGallerySelectionBoardId = (values: Record<string, unknown>): string =>
  getGallerySearchScope(values) === 'all'
    ? ALL_READABLE_BOARDS_ID
    : (getGalleryRawSelectedBoardId(values) ?? getGalleryProjectBoardId(values) ?? 'none');

/** Date buckets defer to the project; explicit none stays Uncategorized. */
export const getGalleryDestinationBoardId = (values: Record<string, unknown>): string | null => {
  const selectedBoardId = getGalleryRawSelectedBoardId(values);

  return selectedBoardId !== null && !isDateBoardId(selectedBoardId)
    ? selectedBoardId
    : getGalleryProjectBoardId(values);
};

export const getGalleryDefaultDestinationBoardId = (values: Record<string, unknown>): string =>
  getGalleryProjectBoardId(values) ?? 'none';

/** The selected board only while `followSelectedBoard` is on; a board `canAddTo` refuses sends results to the inbox. */
export const getGalleryAutoAddBoardId = (
  values: ResolvedValues,
  canAddTo: (boardId: string) => boolean
): string | null => {
  const { autoAddBoardId, followSelectedBoard } = getGallerySettings(values);
  const target = followSelectedBoard
    ? getGalleryDestinationBoardId(values)
    : (autoAddBoardId ?? getGalleryDefaultDestinationBoardId(values));

  return target !== null && target !== 'none' && !canAddTo(target)
    ? (getGalleryProjectBoardId(values) ?? 'none')
    : target;
};

/** Before any refusal, as the panel and destination control show it. */
export const getGalleryIntendedDestinationId = (
  view: Pick<GalleryStateView, 'projectBoardId' | 'selectedBoardId' | 'settings'>
): string => {
  const { autoAddBoardId, followSelectedBoard } = view.settings;

  if (followSelectedBoard) {
    return isDateBoardId(view.selectedBoardId) ? (view.projectBoardId ?? 'none') : view.selectedBoardId;
  }

  return autoAddBoardId ?? view.projectBoardId ?? 'none';
};

/** An empty board list means loading, so defer resolution. */
export const resolveGallerySelectedBoardId = (
  { projectBoardId, selectedBoardId }: { projectBoardId: string | null; selectedBoardId: string | null },
  backendBoards: GalleryBoard[]
): string => {
  if (backendBoards.length === 0) {
    return selectedBoardId ?? 'none';
  }

  if (selectedBoardId !== null && backendBoards.some((board) => board.id === selectedBoardId)) {
    return selectedBoardId;
  }

  if (projectBoardId !== null && backendBoards.some((board) => board.id === projectBoardId)) {
    return projectBoardId;
  }

  return 'none';
};

export const getGallerySelectedBoardId = (values: Record<string, unknown>, backendBoards: GalleryBoard[]): string =>
  resolveGallerySelectedBoardId(
    { projectBoardId: getGalleryProjectBoardId(values), selectedBoardId: getGalleryRawSelectedBoardId(values) },
    backendBoards
  );

export const getGalleryListingBoardId = (values: Record<string, unknown>, backendBoards: GalleryBoard[]): string =>
  getGallerySearchScope(values) === 'all' ? ALL_READABLE_BOARDS_ID : getGallerySelectedBoardId(values, backendBoards);

export const getGalleryPage = (values: Record<string, unknown>): number =>
  typeof values.galleryPage === 'number' && Number.isFinite(values.galleryPage)
    ? Math.max(0, Math.floor(values.galleryPage))
    : 0;

export const getGallerySelectedImagePage = (values: Record<string, unknown>): number =>
  typeof values.selectedImagePage === 'number' && Number.isFinite(values.selectedImagePage)
    ? Math.max(0, Math.floor(values.selectedImagePage))
    : getGalleryPage(values);

export interface GallerySelectedImageQuery {
  boardId: string;
  galleryView: GalleryView;
  imageOrderDir: GalleryOrderDir;
  /** Null lists both; stamps written before the field existed read as null, never as the current filter. */
  mediaKind: GalleryItemKind | null;
  page: number;
  paginationMode: 'infinite' | 'paginated';
  searchTerm: string;
  /** One made outside the Gallery, such as a search pick: navigates its item's own board, unranked. */
  itemBoard: boolean;
  semanticKey: string | null;
  starredOnly: boolean;
}

export const getGallerySelectedImageQuery = (values: ResolvedValues): GallerySelectedImageQuery => {
  const query =
    values.selectedImageQuery && typeof values.selectedImageQuery === 'object'
      ? (values.selectedImageQuery as Partial<GallerySelectedImageQuery>)
      : null;
  const settings = getGallerySettings(values);

  return {
    boardId: query && typeof query.boardId === 'string' ? query.boardId : getGallerySelectionBoardId(values),
    galleryView:
      query?.galleryView === 'assets' || query?.galleryView === 'images'
        ? query.galleryView
        : values.galleryView === 'assets'
          ? 'assets'
          : 'images',
    imageOrderDir:
      query?.imageOrderDir === 'ASC' || query?.imageOrderDir === 'DESC' ? query.imageOrderDir : settings.imageOrderDir,
    mediaKind: query?.mediaKind === 'image' || query?.mediaKind === 'video' ? query.mediaKind : null,
    page:
      query && typeof query.page === 'number' && Number.isFinite(query.page)
        ? Math.max(0, Math.floor(query.page))
        : getGallerySelectedImagePage(values),
    paginationMode:
      query?.paginationMode === 'infinite' || query?.paginationMode === 'paginated'
        ? query.paginationMode
        : settings.paginationMode,
    itemBoard: query?.itemBoard === true,
    searchTerm: query && typeof query.searchTerm === 'string' ? query.searchTerm : String(values.searchTerm ?? ''),
    semanticKey: query && typeof query.semanticKey === 'string' && query.semanticKey ? query.semanticKey : null,
    starredOnly: query && typeof query.starredOnly === 'boolean' ? query.starredOnly : getGalleryStarredOnly(values),
  };
};

/** A selection stamped elsewhere is not this listing's, though it persists across those switches. */
export const isGallerySelectionInScope = (
  selectedImageQuery: GallerySelectedImageQuery,
  scope: Pick<GalleryStateView, 'galleryView' | 'listingBoardId' | 'searchTerm' | 'settings' | 'starredOnly'>
): boolean =>
  selectedImageQuery.boardId === scope.listingBoardId &&
  selectedImageQuery.galleryView === scope.galleryView &&
  selectedImageQuery.imageOrderDir === scope.settings.imageOrderDir &&
  selectedImageQuery.mediaKind === (toGalleryMediaKind(scope.settings.mediaFilter) ?? null) &&
  selectedImageQuery.searchTerm === scope.searchTerm &&
  selectedImageQuery.starredOnly === scope.starredOnly;

export const getGalleryTotalImages = (values: Record<string, unknown>): number | null =>
  typeof values.galleryTotalImages === 'number' && Number.isFinite(values.galleryTotalImages)
    ? Math.max(0, values.galleryTotalImages)
    : null;

export const getGalleryProjectBoardId = (values: Record<string, unknown>): string | null =>
  typeof values.projectBoardId === 'string' ? values.projectBoardId : null;

/** The existing project field also accepts the canonical video item shape. */
export const getGalleryCompareItem = (values: Record<string, unknown>): GalleryItem | null =>
  getSelectedGalleryItemFromValues({
    selectedBoardId: values.selectedBoardId,
    selectedImage: values.compareImage,
    selectedImageName: null,
  });

export const getGalleryCompareImage = (values: Record<string, unknown>): GalleryImage | null => {
  const item = getGalleryCompareItem(values);

  return item?.kind === 'image' ? galleryImageItemToGalleryImage(item) : null;
};

export const getGalleryAnchoredWindowPage = (values: ResolvedValues): number => {
  const page = getGalleryPage(values);

  return getGallerySettings(values).paginationMode === 'infinite' && page > 0 ? page : 0;
};

/**
 * Derived from Query state on every render, never stored. A retry in flight stays `error` so Retry keeps focus;
 * `stale-error` and `more-error` show earlier results or pages.
 */
export type GalleryReadStatus = 'empty' | 'error' | 'loading' | 'more-error' | 'ready' | 'stale-error';

export interface GalleryReadState {
  status: GalleryReadStatus;
  /** Null while a retry is in flight. */
  error: Error | null;
  isRetrying: boolean;
  retry: () => Promise<void>;
}

export interface GalleryQueryFacts {
  /** Placeholder data carried over from another scope does not count. */
  hasData: boolean;
  isError: boolean;
  /** Survives the reset a refetch applies to a query without data, so a retrying failure still reads as failed. */
  errorUpdateCount: number;
  isFetchNextPageError?: boolean;
}

const isFailedWithoutData = (facts: GalleryQueryFacts): boolean => facts.isError || facts.errorUpdateCount > 0;

export const getGalleryReadStatus = (facts: GalleryQueryFacts, itemCount: number): GalleryReadStatus => {
  if (!facts.hasData) {
    return isFailedWithoutData(facts) ? 'error' : 'loading';
  }

  if (facts.isFetchNextPageError) {
    return 'more-error';
  }

  if (facts.isError) {
    return 'stale-error';
  }

  return itemCount === 0 ? 'empty' : 'ready';
};

/** Recents ride along while the scope loads or has data, never for a failed load. */
export const getGalleryListing = (
  facts: GalleryQueryFacts,
  scopedItems: GalleryItem[]
): { items: GalleryItem[] | null; status: GalleryReadStatus } => {
  const status = getGalleryReadStatus(facts, scopedItems.length);

  if (status === 'error' || (status === 'loading' && scopedItems.length === 0)) {
    return { items: null, status };
  }

  return { items: scopedItems, status };
};

export const getGalleryStateView = (
  values: ResolvedValues,
  backendBoards: GalleryBoard[],
  backendItems: GalleryItem[] | null,
  starredStripItems: readonly GalleryItem[] = []
): GalleryStateView => {
  // Unfiltered recents would read as this board's.
  const items = backendItems ?? [];
  const selectedItem = getSelectedGalleryItemFromValues(values);
  const persistedSelectedItemKey =
    typeof values.selectedImageName === 'string'
      ? (getPersistedSelectedGalleryItemKeys({ selectedImageName: values.selectedImageName })[0] ?? null)
      : selectedItem
        ? toGalleryItemKey(selectedItem)
        : null;
  const isVisible = (item: GalleryItem) => toGalleryItemKey(item) === persistedSelectedItemKey;
  const visibleSelectedItemKey =
    persistedSelectedItemKey && (items.some(isVisible) || starredStripItems.some(isVisible))
      ? persistedSelectedItemKey
      : null;
  const selectedItemKeys = getPersistedSelectedGalleryItemKeys(values);
  const galleryView = getGalleryView(values);
  const settings = getGallerySettings(values);
  const searchTerm = getGallerySearchTerm(values);
  const starredOnly = getGalleryStarredOnly(values);
  const boards = backendBoards.length
    ? backendBoards
    : [
        {
          ...UNCATEGORIZED_BOARD,
          assetVideoCount: items.filter((item) => item.kind === 'video' && item.category !== 'general').length,
          imageCount: items.filter((item) => item.kind === 'image' && item.category === 'general').length,
          projectId: null,
          videoCount: items.filter((item) => item.kind === 'video').length,
        },
      ];
  const selectedBoardId = getGallerySelectedBoardId(values, backendBoards);
  const searchScope = getGallerySearchScope(values);
  const listingBoardId = searchScope === 'all' ? ALL_READABLE_BOARDS_ID : selectedBoardId;
  const compareItem = getGalleryCompareItem(values);
  const compareImageKey = compareItem ? toGalleryItemKey(compareItem) : null;
  const isComparisonActive =
    compareItem !== null &&
    visibleSelectedItemKey?.startsWith(`${compareItem.kind}:`) === true &&
    compareImageKey !== visibleSelectedItemKey;
  const isCompareArmed = compareImageKey !== null && compareImageKey === visibleSelectedItemKey;
  const semanticImageQuery = getGallerySemanticImageQuery(values);
  const page = getGalleryPage(values);
  const selectedImageQuery = getGallerySelectedImageQuery(values);
  const revealTargetPage =
    settings.paginationMode === 'paginated' &&
    selectedImageQuery.paginationMode === 'paginated' &&
    semanticImageQuery === null &&
    isGallerySelectionInScope(selectedImageQuery, {
      galleryView,
      listingBoardId,
      searchTerm,
      settings,
      starredOnly,
    }) &&
    // A starred item lives in the strip, so the grid must not follow Preview's starred-list page.
    (starredOnly || selectedItem?.starred !== true)
      ? selectedImageQuery.page
      : null;

  return {
    anchoredWindowPage: getGalleryAnchoredWindowPage(values),
    boards,
    compareImageKey,
    galleryView,
    isComparisonActive,
    isCompareArmed,
    isSearchActive: isGallerySearchActive(values),
    items,
    listingBoardId,
    page,
    projectBoardId: getGalleryProjectBoardId(values),
    revealTargetPage,
    searchScope,
    searchTerm,
    selectedBoardId,
    primarySelectedItemKey: persistedSelectedItemKey,
    selectedItemKey: visibleSelectedItemKey,
    selectedItemKeys:
      visibleSelectedItemKey && !selectedItemKeys.includes(visibleSelectedItemKey)
        ? [visibleSelectedItemKey, ...selectedItemKeys]
        : selectedItemKeys,
    selectionStarredOnly: selectedImageQuery.starredOnly,
    semanticImageQuery,
    semanticSearchText: getGallerySemanticSearchText(values),
    settings,
    starredOnly,
  };
};

/** Uploaded videos count in both `videoCount` and `assetVideoCount`. Clamped: parallel reads can disagree briefly. */
export const getBoardCounts = (
  board: GalleryBoard
): { images: number; items: number; media: number; uploads: number; videos: number } => {
  const images = Math.max(0, board.imageCount);
  const videos = Math.max(0, board.videoCount - board.assetVideoCount);
  const uploads = Math.max(0, board.assetCount + board.assetVideoCount);

  return { images, items: images + videos + uploads, media: images + videos, uploads, videos };
};
