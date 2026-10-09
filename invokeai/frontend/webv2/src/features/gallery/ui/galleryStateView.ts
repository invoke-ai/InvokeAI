import type { GalleryBoard, GalleryImage, GalleryOrderDir, GalleryView } from '@features/gallery/core/types';

import { isDateBoardId, toGalleryItemKey, type GalleryItem, type GalleryItemKey } from '@features/gallery/core/items';
import {
  getPersistedSelectedGalleryItemKeys,
  getSelectedGalleryImageFromValues,
  getSelectedGalleryItemFromValues,
} from '@features/gallery/core/selection';
import {
  parseGallerySemanticReference,
  type GallerySemanticReference,
} from '@features/gallery/core/semanticImageQuery';
import { GALLERY_AUTO_ADD_FOLLOW, getGallerySettings, type GallerySettings } from '@features/gallery/core/settings';

/** Leave the placeholder name empty; getGalleryBoardLabel localizes it from kind. */
const UNCATEGORIZED_BOARD: GalleryBoard = {
  archived: false,
  assetCount: 0,
  assetVideoCount: 0,
  id: 'none',
  imageCount: 0,
  kind: 'uncategorized',
  name: '',
  projectId: null,
  videoCount: 0,
};

export interface GalleryStateView {
  /**
   * A nonzero infinite anchor prevents scrolling to earlier rows; the surface must explain it and provide a return
   * to the top.
   */
  anchoredWindowPage: number;
  boards: GalleryBoard[];
  compareImageKey: GalleryItemKey | null;
  galleryView: GalleryView;
  /** A compare image is set and differs from the visible image selection. */
  isComparisonActive: boolean;
  /** The current scope's listing only; how it stands (loading, failed, ...) is the listing read state's to say. */
  items: GalleryItem[];
  /** The grid's current page in paginated mode; the window anchor otherwise. */
  page: number;
  projectBoardId: string | null;
  /**
   * The selection's stamped paginated page, when the stamp names the listing
   * the grid is showing; null otherwise. Reveals follow it across pages.
   */
  revealTargetPage: number | null;
  searchTerm: string;
  selectedBoardId: string;
  /** The persisted primary selection, retained while its sparse page is not loaded. */
  primarySelectedItemKey: GalleryItemKey | null;
  selectedItemKey: GalleryItemKey | null;
  selectedItemKeys: GalleryItemKey[];
  /**
   * The selection was made in a starred-only listing, so its members were starred when selected. A selection carried
   * in from another listing says nothing about the star flags of members no page has loaded.
   */
  selectionStarredOnly: boolean;
  /** Active image-similarity query, rendered as a chip in place of the search text. */
  semanticImageQuery: GallerySemanticReference | null;
  /** The semantic field's text while the field is in semantic mode; null in metadata mode. */
  semanticSearchText: string | null;
  settings: GallerySettings;
  /** The listing is restricted to starred items. */
  starredOnly: boolean;
}

export const getGalleryView = (values: Record<string, unknown>): GalleryView =>
  values.galleryView === 'assets' ? 'assets' : 'images';

export const getGallerySearchTerm = (values: Record<string, unknown>): string =>
  typeof values.searchTerm === 'string' ? values.searchTerm : '';

/** The starred-only listing filter; a session value kept beside `searchTerm`. */
export const getGalleryStarredOnly = (values: Record<string, unknown>): boolean => values.starredOnly === true;

export const getGallerySemanticImageQuery = (values: Record<string, unknown>): GallerySemanticReference | null =>
  parseGallerySemanticReference(values.semanticImageQuery);

/** Semantic mode is the presence of its text: null means the field searches metadata. */
export const getGallerySemanticSearchText = (values: Record<string, unknown>): string | null =>
  typeof values.semanticSearchText === 'string' ? values.semanticSearchText : null;

/** The saved board choice as persisted, before any resolution against loaded boards. */
export const getGalleryRawSelectedBoardId = (values: Record<string, unknown>): string | null =>
  typeof values.selectedBoardId === 'string' ? values.selectedBoardId : null;

/**
 * The board a selection made in the grid is stamped with: the saved choice, else the project board the grid falls
 * back to showing. Stamping Uncategorized instead points Preview at a listing the selection is not in.
 */
export const getGallerySelectionBoardId = (values: Record<string, unknown>): string =>
  getGalleryRawSelectedBoardId(values) ?? getGalleryProjectBoardId(values) ?? 'none';

/**
 * Use the chosen destination or project board; date buckets defer to the project, while explicit none remains
 * Uncategorized.
 */
export const getGalleryDestinationBoardId = (values: Record<string, unknown>): string | null => {
  const selectedBoardId = getGalleryRawSelectedBoardId(values);

  return selectedBoardId !== null && !isDateBoardId(selectedBoardId)
    ? selectedBoardId
    : getGalleryProjectBoardId(values);
};

/** Where results without a board of their own go: the auto-add board, or the destination above while following. */
export const getGalleryAutoAddBoardId = (values: Record<string, unknown>): string | null => {
  const { autoAddBoardId } = getGallerySettings(values);

  return autoAddBoardId === GALLERY_AUTO_ADD_FOLLOW ? getGalleryDestinationBoardId(values) : autoAddBoardId;
};

/**
 * Preserve valid destinations, otherwise use the project board. An empty board list means loading, so defer
 * resolution.
 */
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
  page: number;
  paginationMode: 'infinite' | 'paginated';
  searchTerm: string;
  /** The selection navigates its item's own board, unranked: one made outside the Gallery, such as a search pick. */
  itemBoard: boolean;
  /** Ranking identity for a semantic result page; null for ordinary listings and legacy state. */
  semanticKey: string | null;
  starredOnly: boolean;
}

export const getGallerySelectedImageQuery = (values: Record<string, unknown>): GallerySelectedImageQuery => {
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

/**
 * The selection was made in the listing the view shows: its stamp names the same board, view, order, search and
 * starred filter. A selection stamped elsewhere (another board, or before a search or filter changed) is not this
 * listing's, though it persists across those switches.
 */
export const isGallerySelectionInScope = (
  selectedImageQuery: GallerySelectedImageQuery,
  scope: Pick<GalleryStateView, 'galleryView' | 'searchTerm' | 'selectedBoardId' | 'settings' | 'starredOnly'>
): boolean =>
  selectedImageQuery.boardId === scope.selectedBoardId &&
  selectedImageQuery.galleryView === scope.galleryView &&
  selectedImageQuery.imageOrderDir === scope.settings.imageOrderDir &&
  selectedImageQuery.searchTerm === scope.searchTerm &&
  selectedImageQuery.starredOnly === scope.starredOnly;

export const getGalleryTotalImages = (values: Record<string, unknown>): number | null =>
  typeof values.galleryTotalImages === 'number' && Number.isFinite(values.galleryTotalImages)
    ? Math.max(0, values.galleryTotalImages)
    : null;

export const getGalleryProjectBoardId = (values: Record<string, unknown>): string | null =>
  typeof values.projectBoardId === 'string' ? values.projectBoardId : null;

export const getGalleryCompareImage = (values: Record<string, unknown>): GalleryImage | null =>
  getSelectedGalleryImageFromValues({
    selectedBoardId: values.selectedBoardId,
    selectedImage: values.compareImage,
    selectedImageName: null,
  });

/** The infinite window's anchor page; 0 whenever the window covers the top of the listing. */
export const getGalleryAnchoredWindowPage = (values: Record<string, unknown>): number => {
  const page = getGalleryPage(values);

  return getGallerySettings(values).paginationMode === 'infinite' && page > 0 ? page : 0;
};

/**
 * How one Gallery read model stands for the scope its query is keyed on (board, search, view, page). Derived from
 * Query state on every render, never stored.
 *
 * - `loading`: nothing for this scope yet, and no failure on record.
 * - `ready` / `empty`: this scope's latest request succeeded, with or without items.
 * - `error`: this scope has nothing to show and its latest request failed. A retry in flight stays here, so the
 *   Retry control keeps focus instead of flashing back to a loading state.
 * - `stale-error`: this scope's earlier results are shown; refreshing them failed.
 * - `more-error`: this scope's earlier pages are shown; the next page failed.
 */
export type GalleryReadStatus = 'empty' | 'error' | 'loading' | 'more-error' | 'ready' | 'stale-error';

/** A read model's standing plus its Query-backed recovery. */
export interface GalleryReadState {
  status: GalleryReadStatus;
  /** The failure behind a failed status, for its detail line; null while a retry is in flight. */
  error: Error | null;
  /** A failed scope is being fetched again; Retry shows busy without unmounting. */
  isRetrying: boolean;
  /** Re-runs this scope's failed request through Query. */
  retry: () => Promise<void>;
}

/** The Query facts a read status derives from, for the query keyed on the current scope. */
export interface GalleryQueryFacts {
  /** Data for this exact scope. Placeholder data carried over from another scope does not count. */
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

/**
 * The listing for the current scope. `scopedItems` is the backend window merged with recents already filtered to
 * this scope (only the recents while nothing has loaded); they may ride along while the scope loads or has data,
 * but never stand in for a failed load.
 */
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

/** Starred selections remain visible in the pinned strip rather than the unstarred listing. */
export const getGalleryStateView = (
  values: Record<string, unknown>,
  backendBoards: GalleryBoard[],
  backendItems: GalleryItem[] | null,
  starredStripItems: readonly GalleryItem[] = []
): GalleryStateView => {
  // Nothing stands in for a listing the current scope does not have: unfiltered recents would read as this board's.
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
  const compareImage = getGalleryCompareImage(values);
  const compareImageKey = compareImage ? toGalleryItemKey({ kind: 'image', name: compareImage.imageName }) : null;
  const isComparisonActive =
    visibleSelectedItemKey?.startsWith('image:') === true &&
    compareImageKey !== null &&
    compareImageKey !== visibleSelectedItemKey;
  const semanticImageQuery = getGallerySemanticImageQuery(values);
  const page = getGalleryPage(values);
  const selectedImageQuery = getGallerySelectedImageQuery(values);
  const revealTargetPage =
    settings.paginationMode === 'paginated' &&
    selectedImageQuery.paginationMode === 'paginated' &&
    semanticImageQuery === null &&
    isGallerySelectionInScope(selectedImageQuery, {
      galleryView,
      searchTerm,
      selectedBoardId,
      settings,
      starredOnly,
    }) &&
    // A starred item lives in the strip, never on a page of the unstarred
    // listing; Preview stamps its starred-list page, which the grid must not follow.
    (starredOnly || selectedItem?.starred !== true)
      ? selectedImageQuery.page
      : null;

  return {
    anchoredWindowPage: getGalleryAnchoredWindowPage(values),
    boards,
    compareImageKey,
    galleryView,
    isComparisonActive,
    items,
    page,
    projectBoardId: getGalleryProjectBoardId(values),
    revealTargetPage,
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

export const getBoardCounts = (
  board: GalleryBoard
): { assetCount: number; assetVideoCount: number; imageCount: number; videoCount: number } => ({
  assetCount: board.assetCount,
  assetVideoCount: board.assetVideoCount,
  imageCount: board.imageCount,
  videoCount: board.videoCount,
});
