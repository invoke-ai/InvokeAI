import { Box, chakra, Flex, HStack, Icon, ScrollArea, Spinner, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey, type GalleryItem, type GalleryItemKey } from '@features/gallery/core/items';
import {
  getGalleryRevealRequest,
  getGallerySessionNavigationKey,
  getSelectedGalleryItemFromValues,
  subscribeGalleryRevealRequests,
  type GalleryNavigationEntry,
  type GalleryRevealRequest,
} from '@features/gallery/core/selection';
import { isDateBoardId } from '@features/gallery/data/backend';
import {
  GALLERY_PAGE_SIZE,
  imageIndexAvailabilityOptions,
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { Button, DropZone } from '@platform/ui';
import { useQuery } from '@tanstack/react-query';
import { ChevronRightIcon, StarIcon, UploadIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type DragEvent,
  type FocusEvent,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { flushSync } from 'react-dom';
import { defaultRangeExtractor, useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GallerySparsePageState } from './useGalleryData';

import {
  buildGalleryGridRows,
  buildSparseGalleryNavigationEntries,
  chunkGalleryCellsIntoRows,
  GALLERY_GRID_GAP_PX,
  GALLERY_PINNED_FOOTER_PX,
  GALLERY_STARRED_HEADER_HEIGHT_PX,
  getGalleryCellSizePx,
  getGalleryColumnCount,
  getGalleryGridRowIndexForItemKey,
  getGallerySparseRowIndexForItemKey,
  getGallerySparseRowKey,
  getGallerySparseSelectionPages,
  getGallerySparseSlotKey,
  getGalleryPinnedHeightPx,
  getGalleryProgressLayout,
  getGalleryStarredLayout,
  getGalleryStarredStripItems,
} from './galleryGridLayout';
import { focusVisibleOperable, GalleryLoadErrorState, GalleryLoadNotice, GalleryRetryButton } from './GalleryLoadError';
import { GalleryProgressSection } from './GalleryProgressSection';
import { getGallerySelectedImageQuery, isGallerySelectionInScope } from './galleryStateView';
import { GALLERY_TAB_STOP_SELECTOR, GalleryThumbnailCell } from './GalleryThumbnail';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget, type GalleryStarredStrip } from './GalleryWidgetContext';
import {
  useGalleryGridHotkeys,
  type GalleryNavigationMode,
  type GalleryUnloadedNavigationRequest,
} from './useGalleryGridHotkeys';
import { useGalleryGridSelection } from './useGalleryGridSelection';
import { useGalleryUploadInput } from './useGalleryUploadInput';

type VirtualRange = Parameters<typeof defaultRangeExtractor>[0];

/**
 * Seeds the measured width per region so remounting the gallery in a placement
 * it has already been shown in does not paint one frame of fallback-sized
 * tiles before the ResizeObserver reports.
 */
const viewportWidthCache = new Map<string, number>();
const STARRED_TRIGGER_HOVER_STYLES = { color: 'fg' } as const;
const EMPTY_GALLERY_ITEMS: GalleryItem[] = [];

// Module-scoped so a grid remount cannot replay an already-followed reveal.
let lastPageFollowedRevealToken = 0;

const dragEventContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

const GalleryPageError = ({
  onFocusLost,
  compact = false,
  pageOffset,
  pageState,
}: {
  onFocusLost: () => void;
  compact?: boolean;
  pageOffset: number;
  pageState: GallerySparsePageState;
}) => {
  const { t } = useTranslation();
  const isMoreError = pageOffset > 0;
  const read = useMemo(
    () => ({
      isRetrying: pageState.isLoading,
      retry: async () => {
        await pageState.retry();
      },
    }),
    [pageState]
  );
  const message = t(isMoreError ? 'widgets.gallery.listingLoadMoreFailed' : 'widgets.gallery.listingRefreshFailed');

  if (compact) {
    return (
      <>
        <VisuallyHidden>
          {message} {pageState.error?.message}
        </VisuallyHidden>
        <GalleryRetryButton
          aria-label={t(isMoreError ? 'widgets.gallery.retryLoadingMoreItems' : 'widgets.gallery.retryLoadingItems')}
          color="fg"
          flagsFailure
          read={read}
          size="xs"
          variant="ghost"
          onFocusLost={onFocusLost}
        />
      </>
    );
  }

  return (
    <Stack align="center" gap="1" maxW="full" px="1">
      <Text color="fg.muted" fontSize="xs" lineClamp={2} textAlign="center">
        {message}
      </Text>
      <GalleryRetryButton
        aria-label={t(isMoreError ? 'widgets.gallery.retryLoadingMoreItems' : 'widgets.gallery.retryLoadingItems')}
        color="fg"
        flagsFailure
        read={read}
        size="sm"
        variant="ghost"
        onFocusLost={onFocusLost}
      />
    </Stack>
  );
};

const GallerySparseSlot = ({
  cellSizePx,
  onFocusLost,
  pageOffset,
  pageState,
  showPageStatus,
}: {
  cellSizePx: number;
  onFocusLost: () => void;
  pageOffset: number;
  pageState: GallerySparsePageState | undefined;
  showPageStatus: boolean;
}) => {
  const { t } = useTranslation();
  const error = pageState?.error;
  const state = error && showPageStatus ? 'error' : pageState?.isLoading && showPageStatus ? 'loading' : 'empty';

  return (
    <Box
      aria-busy={state === 'loading' || undefined}
      aspectRatio="1"
      bg={state === 'loading' ? 'bg.subtle' : undefined}
      data-gallery-slot-state={state}
      display="flex"
      h={`${cellSizePx}px`}
      alignItems="center"
      justifyContent="center"
      minW="0"
      overflow="hidden"
      role="listitem"
      rounded="sm"
      aria-label={
        state === 'error'
          ? t(pageOffset > 0 ? 'widgets.gallery.listingLoadMoreFailed' : 'widgets.gallery.listingRefreshFailed')
          : undefined
      }
    >
      {state === 'loading' ? (
        <Stack align="center" aria-label={t('widgets.gallery.loadingBackendGallery')} gap="1" role="status">
          <Spinner aria-hidden="true" size="md" />
          <Text color="fg.muted" fontSize="xs" lineClamp={1}>
            {t('widgets.gallery.loadingBackendGallery')}
          </Text>
        </Stack>
      ) : state === 'error' && pageState ? (
        <GalleryPageError compact onFocusLost={onFocusLost} pageOffset={pageOffset} pageState={pageState} />
      ) : null}
    </Box>
  );
};

/** Show all appears only when starred items exceed the strip and activates the starred-only listing. */
const GalleryStarredSectionHeader = ({
  isOpen,
  onFocusLost,
  onShowAll,
  onToggle,
  shownCount,
  state,
  total,
}: {
  isOpen: boolean;
  onFocusLost: () => void;
  onShowAll: () => void;
  onToggle: () => void;
  shownCount: number;
  state: GalleryStarredStrip['state'];
  total: number;
}) => {
  const { t } = useTranslation();

  return (
    <Flex align="center" gap="1" h={`${GALLERY_STARRED_HEADER_HEIGHT_PX}px`} px="1" w="full">
      <chakra.button
        aria-expanded={isOpen}
        aria-label={t(isOpen ? 'widgets.gallery.collapseStarredItems' : 'widgets.gallery.expandStarredItems')}
        alignItems="center"
        color="fg.muted"
        display="flex"
        flex="1"
        gap="1"
        minW="0"
        transition="color var(--wb-motion-duration-fast) ease"
        type="button"
        _hover={STARRED_TRIGGER_HOVER_STYLES}
        onClick={onToggle}
      >
        <Icon
          as={ChevronRightIcon}
          boxSize="3"
          transform={isOpen ? 'rotate(90deg)' : undefined}
          transition="transform var(--wb-motion-duration-medium) ease"
        />
        <Icon as={StarIcon} boxSize="3" fill="currentColor" />
        <HStack gap="1" minW="0">
          <Text
            as="span"
            fontSize="xs"
            fontWeight="600"
            letterSpacing="wide"
            lineHeight="1"
            textTransform="uppercase"
            truncate
          >
            {t('widgets.gallery.starredItems')}
          </Text>
          <Text as="span" color="currentColor" fontSize="xs" fontVariantNumeric="tabular-nums" lineHeight="1">
            {total}
          </Text>
        </HStack>
      </chakra.button>
      {state.status === 'stale-error' ? (
        // The strip keeps its earlier cells; the header carries the failed refresh and its recovery.
        <GalleryRetryButton
          aria-label={t('widgets.gallery.retryRefreshingStarredItems')}
          color="fg.error"
          flagsFailure
          read={state}
          size="xs"
          variant="ghost"
          onFocusLost={onFocusLost}
        />
      ) : null}
      {total > shownCount ? (
        <Button
          aria-label={t('widgets.gallery.showAllStarredItems')}
          color="fg.muted"
          flexShrink={0}
          size="sm"
          variant="ghost"
          onClick={onShowAll}
        >
          {t('widgets.gallery.showAllStarred')}
        </Button>
      ) : null}
    </Flex>
  );
};

/**
 * Render the bounded starred strip outside virtualization while preserving its place in cross-section keyboard
 * navigation.
 */
const GalleryStarredSection = ({
  cells,
  cellSizePx,
  columnCount,
  isOpen,
  renderCell,
  state,
  total,
  onFocusLost,
  onShowAll,
  onToggle,
}: {
  cells: GalleryItem[];
  /** Rows take the listing's pitch explicitly, so the pinned block measures what the layout computed. */
  cellSizePx: number;
  columnCount: number;
  isOpen: boolean;
  renderCell: (item: GalleryItem) => ReactNode;
  state: GalleryStarredStrip['state'];
  total: number;
  onFocusLost: () => void;
  onShowAll: () => void;
  onToggle: () => void;
}) => {
  const { t } = useTranslation();
  const rows = useMemo(() => chunkGalleryCellsIntoRows(cells, columnCount, 'starred'), [cells, columnCount]);

  return (
    <Box>
      <GalleryStarredSectionHeader
        isOpen={isOpen}
        shownCount={isOpen ? cells.length : 0}
        state={state}
        total={total}
        onFocusLost={onFocusLost}
        onShowAll={onShowAll}
        onToggle={onToggle}
      />
      {isOpen ? (
        <Box aria-label={t('widgets.gallery.starredItems')} role="list">
          {rows.map((row) => (
            <Box
              key={row.key}
              data-gallery-section="starred"
              display="grid"
              gap={`${GALLERY_GRID_GAP_PX}px`}
              gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
              h={`${cellSizePx}px`}
              mb={`${GALLERY_GRID_GAP_PX}px`}
              role="presentation"
              w="full"
            >
              {row.cells.map(renderCell)}
            </Box>
          ))}
        </Box>
      ) : null}
    </Box>
  );
};

/** A tile's own select button, not the star toggle layered over it. */
const TILE_BUTTON_SELECTOR = '[role="listitem"] button[aria-pressed]';

const NO_ITEMS: GalleryItem[] = [];
const GRID_OVERSCAN_ROWS = 4;

/** Where keyboard focus last was among the thumbnails: the scope and position outlive the tile itself. */
interface FocusedTile {
  filter: GalleryItemsFilter;
  index: number;
  key: GalleryItemKey;
}

/** The rows in view plus the one keyboard focus needs wherever the grid scrolls; -1 keeps nothing. */
const extractRangeKeeping = (range: VirtualRange, keptRow: number): number[] => {
  const indexes = defaultRangeExtractor(range);

  return keptRow < 0 || keptRow >= range.count || indexes.includes(keptRow)
    ? indexes
    : [...indexes, keptRow].sort((a, b) => a - b);
};

/** The item of the thumbnail tile holding `element`, whether its select button or the star toggle over it. */
const getTileItemKey = (element: Element | null): GalleryItemKey | null =>
  (element
    ?.closest('[role="listitem"]')
    ?.querySelector('[data-gallery-item-key]')
    ?.getAttribute('data-gallery-item-key') ?? null) as GalleryItemKey | null;

/** The tile that shows a navigation entry: a thumbnail, or an in-progress session. */
const findEntryTile = (viewport: HTMLElement, entry: GalleryNavigationEntry): HTMLElement | null =>
  entry.kind === 'session'
    ? viewport.querySelector<HTMLElement>(`[data-gallery-session-id="${CSS.escape(entry.id)}"]`)
    : entry.kind === 'item'
      ? viewport.querySelector<HTMLElement>(`[data-gallery-item-key="${CSS.escape(toGalleryItemKey(entry.item))}"]`)
      : null;

/** Prefers a visible tile; an empty grid offers whatever else it shows (its upload target, the strip header). */
const focusVisibleGridContent = (viewport: HTMLElement | null, edge: 'first' | 'last') => {
  if (!focusVisibleOperable(viewport, { edge, selector: TILE_BUTTON_SELECTOR })) {
    focusVisibleOperable(viewport, { edge });
  }
};

/** Measure viewport width for columns so both layouts share the same grid. */
export const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const {
    actions,
    filter,
    gallery,
    isWindowTruncated,
    itemActions,
    listing,
    region,
    setVisibleRange,
    sparseListing,
    starredStrip,
  } = useGalleryWidget();
  const {
    gallery: galleryCommands,
    galleryValues,
    getItemLabel,
    ImageContextMenu,
    followedProgressSessionId,
    progressSessions,
  } = useGalleryUi();
  const { data: indexAvailability } = useQuery(imageIndexAvailabilityOptions());
  const getReadyItemLabel = indexAvailability?.state === 'ready' ? getItemLabel : null;
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(region) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const {
    imageDensityPercent,
    paginationMode,
    progressSectionCollapsed,
    showImageDimensions,
    showPendingItems,
    starredSectionCollapsed,
    thumbnailFit,
  } = gallery.settings;
  const isStarredOpen = !starredSectionCollapsed;
  const usesSparseListing = sparseListing !== undefined;
  const isSparsePaginated = usesSparseListing && paginationMode === 'paginated';
  const sparsePageOffset = isSparsePaginated ? gallery.page * GALLERY_PAGE_SIZE : 0;
  const sparseSelectionPages = useMemo(
    () =>
      sparseListing
        ? getGallerySparseSelectionPages({ itemSlots: sparseListing.itemSlots, pageOffset: sparsePageOffset })
        : new Map<GalleryItemKey, number>(),
    [sparseListing, sparsePageOffset]
  );
  const getSelectionPage = useCallback(
    (item: GalleryItem) => sparseSelectionPages.get(toGalleryItemKey(item)),
    [sparseSelectionPages]
  );

  const {
    actionSelectionRefs,
    activeContextMenuTarget,
    getDragItems,
    handleCloseContextMenu,
    handleThumbnailClick,
    handleThumbnailContextMenu,
    loadedItems,
    selectedItemKeys,
    selectItemRange,
    syncRangeInteractionContext,
    toggleItem,
  } = useGalleryGridSelection({ getSelectionPage });

  const columnCount = getGalleryColumnCount({ imageDensityPercent, widthPx: viewportWidth });
  const sparseRecentItems = usesSparseListing ? sparseListing.recentItems : EMPTY_GALLERY_ITEMS;
  const sparseRecentRowCount = Math.ceil(sparseRecentItems.length / columnCount);
  const sparseRecentAtTop = usesSparseListing && !isSparsePaginated && gallery.settings.imageOrderDir === 'DESC';
  const sparseFilterIdentity = JSON.stringify(filter);
  const leadingRecentRows = sparseRecentAtTop ? sparseRecentRowCount : 0;
  const sparseBackendItemCount = usesSparseListing
    ? isSparsePaginated
      ? Math.min(GALLERY_PAGE_SIZE, Math.max(0, (sparseListing.total ?? 0) - sparsePageOffset))
      : (sparseListing.total ?? 0)
    : 0;
  const isFollowingLive = followedProgressSessionId !== null;
  const isComparisonActive = gallery.isComparisonActive && !isFollowingLive;
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const selectedBoardName = selectedBoard
    ? getGalleryBoardLabel(selectedBoard, t)
    : t('widgets.gallery.selectedBoardFallback');
  // The listing is unstarred-only, so a board whose items are all starred
  // still has the strip to show.
  const isEmpty =
    (usesSparseListing ? sparseBackendItemCount === 0 && sparseRecentItems.length === 0 : gallery.items.length === 0) &&
    starredStrip.items.length === 0;
  // A failed strip with no cells to show (none loaded, or an empty earlier result) reports in the header's place.
  const isStarredFailedEmpty =
    (starredStrip.state.status === 'error' || starredStrip.state.status === 'stale-error') &&
    starredStrip.items.length === 0;
  // A ranking that matched nothing is still a search result, never an empty
  // board inviting an upload.
  const hasActiveSearch = gallery.searchTerm.trim() !== '' || gallery.semanticImageQuery !== null;
  const isVirtualBoard = isDateBoardId(gallery.selectedBoardId);

  const rows = useMemo(() => buildGalleryGridRows(gallery.items, columnCount), [columnCount, gallery.items]);
  const starredCells = useMemo(
    () => getGalleryStarredStripItems(starredStrip.items, columnCount),
    [columnCount, starredStrip.items]
  );
  // Exclude collapsed tiles from navigation, but retain hidden starred selections' section identity so arrows can
  // step out.
  const isProgressOpen = showPendingItems && !progressSectionCollapsed;
  const shownStripItems = isStarredOpen ? starredCells : NO_ITEMS;
  // A starred selection no tile shows still belongs to the strip: one under the collapsed disclosure or past the
  // shown rows, which the strip holds, or — as in Preview — one beyond the strip's bound, which the view names no
  // visible key for and only the persisted selection holds (the listing is unstarred). Either way the arrows step
  // from it rather than from the first tile. The persisted selection counts only when made in this listing: one
  // left over from another board, search or filter would put a phantom entry at the end of this strip.
  const hiddenStripSelection = useMemo((): GalleryItem | null => {
    const selectedItem = getSelectedGalleryItemFromValues(galleryValues);
    const selectedKey = gallery.selectedItemKey ?? (selectedItem ? toGalleryItemKey(selectedItem) : null);
    const isSelected = (item: GalleryItem) => toGalleryItemKey(item) === selectedKey;

    if (selectedKey === null || shownStripItems.some(isSelected) || gallery.items.some(isSelected)) {
      return null;
    }

    return (
      starredStrip.items.find(isSelected) ??
      (selectedItem?.starred &&
      isSelected(selectedItem) &&
      isGallerySelectionInScope(getGallerySelectedImageQuery(galleryValues), gallery)
        ? selectedItem
        : null)
    );
  }, [gallery, galleryValues, shownStripItems, starredStrip.items]);
  const navigationSections = useMemo((): GalleryNavigationEntry[][] => {
    const stripEntries: GalleryNavigationEntry[] = shownStripItems.map((item) => ({ item, kind: 'item' }));

    if (hiddenStripSelection) {
      stripEntries.push({ item: hiddenStripSelection, kind: 'item' });
    }

    const listingEntries =
      usesSparseListing && sparseListing
        ? buildSparseGalleryNavigationEntries({
            columnCount,
            includeUnloadedBoundaries: !isSparsePaginated,
            itemSlots: sparseListing.itemSlots,
            pageOffsets: isSparsePaginated ? [0] : [...sparseListing.pageStates.keys()],
            total: isSparsePaginated ? sparseBackendItemCount : sparseListing.total,
          })
        : gallery.items.map((item) => ({ item, kind: 'item' }) as const);
    const recentEntries = sparseRecentItems.map((item) => ({ item, kind: 'item' }) as const);

    return [
      stripEntries,
      isProgressOpen
        ? progressSessions.map((session) => ({
            id: session.id,
            kind: 'session',
            navigable: session.state === 'running',
          }))
        : [],
      ...(sparseRecentAtTop ? [recentEntries, listingEntries] : [listingEntries, recentEntries]),
    ];
  }, [
    gallery.items,
    columnCount,
    isProgressOpen,
    hiddenStripSelection,
    progressSessions,
    sparseListing,
    isSparsePaginated,
    sparseBackendItemCount,
    sparseRecentAtTop,
    sparseRecentItems,
    shownStripItems,
    usesSparseListing,
  ]);
  const cursorKey =
    followedProgressSessionId !== null
      ? getGallerySessionNavigationKey(followedProgressSessionId)
      : gallery.selectedItemKey;
  // Thumbnails in visual order share one Tab stop. Sparse slots must retain their real row positions even when pages
  // are unloaded, and recent results sit at the order-dependent end of that listing.
  const sparseSlotEntries = useMemo(
    () => [...(sparseListing?.itemSlots.entries() ?? [])].sort(([left], [right]) => left - right),
    [sparseListing?.itemSlots]
  );
  const tileKeys = useMemo(() => {
    const items = usesSparseListing
      ? [
          ...shownStripItems,
          ...(sparseRecentAtTop ? sparseRecentItems : []),
          ...sparseSlotEntries.map(([, item]) => item),
          ...(!sparseRecentAtTop ? sparseRecentItems : []),
        ]
      : [...shownStripItems, ...gallery.items];

    return items.map(toGalleryItemKey);
  }, [gallery.items, shownStripItems, sparseRecentAtTop, sparseRecentItems, sparseSlotEntries, usesSparseListing]);
  const [focusedTile, setFocusedTile] = useState<FocusedTile | null>(null);
  const selectedTileIndex = gallery.selectedItemKey === null ? -1 : tileKeys.indexOf(gallery.selectedItemKey);
  const focusedTileIndex = focusedTile === null ? -1 : tileKeys.indexOf(focusedTile.key);
  const tabStopIndex =
    focusedTileIndex >= 0
      ? focusedTileIndex
      : selectedTileIndex >= 0
        ? selectedTileIndex
        : focusedTile?.filter === filter
          ? Math.max(0, Math.min(focusedTile.index, tileKeys.length - 1))
          : 0;
  const tabStopKey = tileKeys[tabStopIndex] ?? null;
  const tabStopItemKey = tabStopKey;
  const tabStopStripIndex =
    tabStopItemKey === null ? -1 : shownStripItems.findIndex((item) => toGalleryItemKey(item) === tabStopItemKey);
  const tabStopRecentIndex =
    usesSparseListing && tabStopItemKey !== null
      ? sparseRecentItems.findIndex((item) => toGalleryItemKey(item) === tabStopItemKey)
      : -1;
  const tabStopSparseIndex =
    usesSparseListing && tabStopItemKey !== null
      ? (sparseSlotEntries.find(([, item]) => toGalleryItemKey(item) === tabStopItemKey)?.[0] ?? -1)
      : -1;
  const sparseBackendRows = Math.ceil(sparseBackendItemCount / columnCount);
  const tabStopRow = usesSparseListing
    ? tabStopStripIndex >= 0
      ? -1
      : tabStopRecentIndex >= 0
        ? (sparseRecentAtTop ? 0 : sparseBackendRows) + Math.floor(tabStopRecentIndex / columnCount)
        : tabStopSparseIndex >= 0
          ? leadingRecentRows + Math.floor(tabStopSparseIndex / columnCount)
          : -1
    : tabStopIndex < shownStripItems.length
      ? -1
      : Math.floor((tabStopIndex - shownStripItems.length) / columnCount);
  // Scrolling never unmounts the Tab stop, which is also the focused tile while the grid holds focus: focus would
  // fall to the document body, and Tab would skip the grid.
  const rangeExtractor = useCallback((range: VirtualRange) => extractRangeKeeping(range, tabStopRow), [tabStopRow]);

  const backendRowCount = usesSparseListing ? Math.ceil(sparseBackendItemCount / columnCount) : rows.length;
  const rowCount = usesSparseListing ? backendRowCount + sparseRecentRowCount : rows.length;
  const cellSizePx = getGalleryCellSizePx({ columnCount, widthPx: viewportWidth });
  const rowHeightPx = cellSizePx + GALLERY_GRID_GAP_PX;
  const estimateRowSize = useCallback(() => rowHeightPx, [rowHeightPx]);
  const getRowKey = useCallback(
    (index: number) => {
      if (!usesSparseListing || !sparseListing) {
        return rows[index]?.key ?? index;
      }

      const isRecentRow = sparseRecentAtTop ? index < sparseRecentRowCount : index >= backendRowCount;

      if (isRecentRow) {
        const recentRow = sparseRecentAtTop ? index : index - backendRowCount;
        const firstRecentItem = sparseRecentItems[recentRow * columnCount];

        return `recent-row:${firstRecentItem ? toGalleryItemKey(firstRecentItem) : recentRow}`;
      }

      const backendRow = index - leadingRecentRows;
      const absoluteBackendRow = Math.floor(sparsePageOffset / columnCount) + backendRow;

      return getGallerySparseRowKey(absoluteBackendRow);
    },
    [
      backendRowCount,
      columnCount,
      leadingRecentRows,
      rows,
      sparsePageOffset,
      sparseListing,
      sparseRecentAtTop,
      sparseRecentItems,
      sparseRecentRowCount,
      usesSparseListing,
    ]
  );
  const getScrollElement = useCallback(() => viewportRef.current, []);

  const progressLayout = getGalleryProgressLayout({
    columns: columnCount,
    tileSize: cellSizePx,
    sessionCount: progressSessions.length,
    visible: showPendingItems,
    collapsed: progressSectionCollapsed,
  });
  const starredLayout = getGalleryStarredLayout({
    collapsed: !isStarredOpen,
    columns: columnCount,
    failed: isStarredFailedEmpty,
    shownCount: starredCells.length,
    tileSize: cellSizePx,
  });
  const pinnedHeight = getGalleryPinnedHeightPx(progressLayout.height, starredLayout.height);
  const currentSparseGeometry = { columnCount, leadingRecentRows, pinnedHeight, rowHeightPx };
  const lastMeasuredSparseGeometryRef = useRef(currentSparseGeometry);
  const sparseScrollAnchorRef = useRef<{
    absoluteIndex: number;
    itemKey: GalleryItemKey;
    viewportOffsetPx: number;
  } | null>(null);
  const sparsePositionSnapshotRef = useRef<{
    filterIdentity: string;
    itemKeys: ReadonlyMap<number, GalleryItemKey>;
    pageOffset: number;
    total: number | null;
  } | null>(null);
  const handleVirtualizerChange = useCallback(
    (instance: {
      getVirtualItems: () => readonly { index: number }[];
      range?: { endIndex: number; startIndex: number } | null;
    }) => {
      if (!usesSparseListing || !sparseListing) {
        return;
      }

      // The range extractor also retains the roving Tab stop, which may be far outside the viewport. Do not let that
      // accessibility row expand the sparse Query subscription or become the scroll anchor.
      const visibleItems = instance
        .getVirtualItems()
        .filter(
          ({ index }) =>
            instance.range === null ||
            instance.range === undefined ||
            (index >= instance.range.startIndex && index <= instance.range.endIndex)
        );
      const viewport = viewportRef.current;
      const previousGeometry = lastMeasuredSparseGeometryRef.current;
      const canCaptureAnchor =
        previousGeometry.columnCount === columnCount &&
        previousGeometry.leadingRecentRows === leadingRecentRows &&
        previousGeometry.pinnedHeight === pinnedHeight &&
        previousGeometry.rowHeightPx === rowHeightPx;

      if (viewport && canCaptureAnchor) {
        for (const visibleRow of visibleItems) {
          const rowTop = pinnedHeight + visibleRow.index * rowHeightPx;

          if (rowTop + rowHeightPx <= viewport.scrollTop) {
            continue;
          }

          const isRecentRow = sparseRecentAtTop
            ? visibleRow.index < sparseRecentRowCount
            : visibleRow.index >= backendRowCount;

          if (isRecentRow) {
            continue;
          }

          const backendRow = visibleRow.index - leadingRecentRows;
          const firstLocalIndex = backendRow * columnCount;
          let capturedAnchor = false;

          for (let column = 0; column < columnCount; column += 1) {
            const localIndex = firstLocalIndex + column;
            const item = sparseListing.itemSlots.get(localIndex);

            if (item) {
              sparseScrollAnchorRef.current = {
                absoluteIndex: sparsePageOffset + localIndex,
                itemKey: toGalleryItemKey(item),
                viewportOffsetPx: rowTop - viewport.scrollTop,
              };
              capturedAnchor = true;
              break;
            }
          }

          if (capturedAnchor) {
            break;
          }
        }
      }

      if (isSparsePaginated || !setVisibleRange) {
        return;
      }

      const firstRow = visibleItems[0]?.index;
      const lastRow = visibleItems[visibleItems.length - 1]?.index;

      if (firstRow === undefined || lastRow === undefined) {
        return;
      }

      const firstBackendRow = Math.min(backendRowCount, Math.max(0, firstRow - leadingRecentRows));
      const afterLastBackendRow = Math.min(backendRowCount, Math.max(0, lastRow - leadingRecentRows + 1));

      if (afterLastBackendRow <= firstBackendRow) {
        // Descending recents precede backend slot zero. Before count discovery, ascending recents keep page zero
        // active; afterward the backend tail page reconciles ascending recent overlays with the authoritative list.
        if (sparseRecentAtTop || sparseListing.total === null) {
          setVisibleRange({ endIndexExclusive: GALLERY_PAGE_SIZE, startIndex: 0 });
        } else {
          const total = sparseListing.total ?? 0;
          const tailPageOffset = total > 0 ? Math.floor((total - 1) / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE : total;
          setVisibleRange({ endIndexExclusive: total, startIndex: tailPageOffset });
        }
        return;
      }

      const startIndex = Math.min(sparseListing.total ?? 0, firstBackendRow * columnCount);
      const endIndexExclusive = Math.min(sparseListing.total ?? 0, afterLastBackendRow * columnCount);

      setVisibleRange({ endIndexExclusive, startIndex });
    },
    [
      backendRowCount,
      columnCount,
      leadingRecentRows,
      isSparsePaginated,
      pinnedHeight,
      rowHeightPx,
      setVisibleRange,
      sparseListing,
      sparsePageOffset,
      sparseRecentAtTop,
      sparseRecentRowCount,
      usesSparseListing,
    ]
  );
  const virtualizer = useVirtualizer({
    count: rowCount,
    scrollMargin: pinnedHeight,
    estimateSize: estimateRowSize,
    getItemKey: getRowKey,
    getScrollElement,
    onChange: handleVirtualizerChange,
    overscan: GRID_OVERSCAN_ROWS,
    rangeExtractor,
  });
  const sparseNavigationGenerationRef = useRef(0);
  const pendingSparseNavigationRef = useRef<{
    anchorKey: GalleryItemKey | null;
    filterIdentity: string;
    index: number;
    mode: GalleryNavigationMode;
    navigationGeneration: number;
    onResolved: (itemKey: GalleryItemKey) => void;
    originItemKey: string | null;
    originFocusKey: string | null;
    shouldRestoreFocus: boolean;
  } | null>(null);
  const pendingSparseFocusRef = useRef<{
    filterIdentity: string;
    itemKey: GalleryItemKey;
    originFocusKey: string;
  } | null>(null);
  const cancelPendingSparseNavigation = useCallback(() => {
    sparseNavigationGenerationRef.current += 1;
    pendingSparseNavigationRef.current = null;
    pendingSparseFocusRef.current = null;
  }, []);
  const requestSparseAbsoluteIndex = useCallback(
    (absoluteIndex: number) => {
      if (!sparseListing || isSparsePaginated || !setVisibleRange) {
        return;
      }

      if (absoluteIndex < 0 || (sparseListing.total !== null && absoluteIndex >= sparseListing.total)) {
        return;
      }

      const pageOffset = Math.floor(absoluteIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
      setVisibleRange({
        endIndexExclusive: Math.min(sparseListing.total ?? Number.POSITIVE_INFINITY, pageOffset + GALLERY_PAGE_SIZE),
        startIndex: pageOffset,
      });
      virtualizer.scrollToIndex(leadingRecentRows + Math.floor(absoluteIndex / columnCount));
    },
    [columnCount, isSparsePaginated, leadingRecentRows, setVisibleRange, sparseListing, virtualizer]
  );
  const handleNavigateToUnloadedSlot = useCallback(
    (index: number, request: GalleryUnloadedNavigationRequest) => {
      const viewport = viewportRef.current;
      const shouldRestoreFocus = Boolean(viewport?.contains(document.activeElement));
      const active = document.activeElement;
      const activeSessionId =
        shouldRestoreFocus && active instanceof HTMLElement
          ? active.closest('[data-gallery-session-id]')?.getAttribute('data-gallery-session-id')
          : null;

      pendingSparseFocusRef.current = null;
      pendingSparseNavigationRef.current = {
        anchorKey: request.anchorKey,
        filterIdentity: sparseFilterIdentity,
        index,
        mode: request.mode,
        navigationGeneration: sparseNavigationGenerationRef.current,
        onResolved: request.onResolved,
        originItemKey: cursorKey,
        originFocusKey: shouldRestoreFocus
          ? activeSessionId
            ? getGallerySessionNavigationKey(activeSessionId)
            : getTileItemKey(active)
          : null,
        shouldRestoreFocus,
      };
      requestSparseAbsoluteIndex(index);
    },
    [cursorKey, requestSparseAbsoluteIndex, sparseFilterIdentity]
  );

  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  // The hotkey callback reads current state when invoked; stabilizing its identity adds no value and interferes
  // with compiler memoization.
  /** Returns whether the item had somewhere to scroll to — a collapsed strip has none. */
  const scrollToItemKey = (itemKey: GalleryItemKey): boolean => {
    if (usesSparseListing && sparseListing) {
      const recentIndex = sparseRecentItems.findIndex((item) => toGalleryItemKey(item) === itemKey);

      if (recentIndex >= 0) {
        const recentRow = Math.floor(recentIndex / columnCount);
        virtualizer.scrollToIndex(sparseRecentAtTop ? recentRow : backendRowCount + recentRow);

        return true;
      }

      const rowIndex = getGallerySparseRowIndexForItemKey(
        sparseListing.itemSlots,
        itemKey,
        columnCount,
        leadingRecentRows
      );

      if (rowIndex >= 0) {
        virtualizer.scrollToIndex(rowIndex);
        return true;
      }
    }

    if (!usesSparseListing) {
      const rowIndex = getGalleryGridRowIndexForItemKey(gallery.items, itemKey, columnCount);

      if (rowIndex >= 0) {
        virtualizer.scrollToIndex(rowIndex);
        return true;
      }
    }

    // Strip cells sit in the pinned block at the top of the scroll content.
    if (isStarredOpen && starredCells.some((item) => toGalleryItemKey(item) === itemKey)) {
      viewportRef.current?.scrollTo({ top: 0 });
      return true;
    }

    return false;
  };
  const scrollToEntry = (entry: GalleryNavigationEntry) => {
    if (entry.kind === 'session') {
      // In-progress tiles sit below the starred strip; scroll only when the target tile's row is out of view.
      const viewport = viewportRef.current;
      const sessionIndex = progressSessions.findIndex((session) => session.id === entry.id);

      if (viewport && sessionIndex >= 0) {
        const rowTop =
          starredLayout.height +
          progressLayout.headerHeight +
          Math.floor(sessionIndex / progressLayout.columns) * progressLayout.rowHeight;
        const rowBottom = rowTop + progressLayout.rowHeight;

        if (rowTop < viewport.scrollTop) {
          viewport.scrollTo({ top: rowTop - progressLayout.headerHeight });
        } else if (rowBottom > viewport.scrollTop + viewport.clientHeight) {
          viewport.scrollTo({ top: rowBottom - viewport.clientHeight });
        }
      }
    } else if (entry.kind === 'item') {
      scrollToItemKey(toGalleryItemKey(entry.item));
    }
  };

  const handleGridFocus = useCallback(
    (event: FocusEvent<HTMLElement>) => {
      const key = getTileItemKey(event.target as HTMLElement);

      if (key) {
        const index = tileKeys.indexOf(key as GalleryItemKey);

        setFocusedTile((current) =>
          current?.key === key && current.index === index && current.filter === filter
            ? current
            : { filter, index, key: key as GalleryItemKey }
        );
      }
    },
    [filter, tileKeys]
  );

  // A tile that leaves the document holding focus (deleted, moved between the strip and the listing, replaced by
  // another board's) takes focus with it. The same item takes it back where it now shows, else the Tab stop, else
  // the grid itself while it has no tiles, so the keys stay with the gallery; a user who already went elsewhere
  // keeps their focus.
  const restoreTileFocus = useCallback((itemKey: GalleryItemKey) => {
    const viewport = viewportRef.current;
    const active = document.activeElement;

    if (viewport && (active === null || active === document.body)) {
      (
        viewport.querySelector<HTMLElement>(`[data-gallery-item-key="${CSS.escape(itemKey)}"]`) ??
        viewport.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR) ??
        viewport
      ).focus({ preventScroll: true });
    }
  }, []);

  /** The tile holding keyboard focus inside the grid: a thumbnail's item key, or a session's navigation key. */
  const getFocusedTileKey = (): string | null => {
    const viewport = viewportRef.current;
    const active = document.activeElement;

    if (!viewport || !(active instanceof HTMLElement) || !viewport.contains(active)) {
      return null;
    }

    const sessionId = active.closest('[data-gallery-session-id]')?.getAttribute('data-gallery-session-id');

    return sessionId ? getGallerySessionNavigationKey(sessionId) : getTileItemKey(active);
  };
  // A followed session the grid does not show (its section collapsed) gives way to the selection, which a starred
  // item beyond the strip's bound is too, though the view names no visible key for it.
  const getCursorCandidates = () => [
    getFocusedTileKey(),
    followedProgressSessionId === null ? null : getGallerySessionNavigationKey(followedProgressSessionId),
    hiddenStripSelection ? toGalleryItemKey(hiddenStripSelection) : gallery.selectedItemKey,
  ];
  /** The first thumbnail in view, where the arrows start when no cursor is on screen. */
  const getFirstVisibleTileKey = (): string | null => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return null;
    }

    const bounds = viewport.getBoundingClientRect();

    // Document order is visual order: the pinned strip, then the rendered listing rows by index.
    for (const tile of viewport.querySelectorAll<HTMLElement>('button[data-gallery-item-key]')) {
      const rect = tile.getBoundingClientRect();

      if (rect.bottom > bounds.top && rect.top < bounds.bottom) {
        return tile.getAttribute('data-gallery-item-key');
      }
    }

    return null;
  };
  const getFocusedItem = () => {
    const key = getTileItemKey(document.activeElement);

    return key && viewportRef.current?.contains(document.activeElement)
      ? (loadedItems.find((item) => toGalleryItemKey(item) === key) ?? null)
      : null;
  };

  /**
   * Arrow keys move keyboard focus to the entry while focus is in the grid; a command run elsewhere leaves focus be,
   * and a focus-only move (no `select`) needs focus in the grid to mean anything.
   */
  const moveToEntry = (entry: GalleryNavigationEntry, select: (() => void) | null) => {
    const viewport = viewportRef.current;

    if (!viewport?.contains(document.activeElement)) {
      if (select) {
        select();
        scrollToEntry(entry);
      }
      return;
    }

    // Committed before focus moves: the target becomes the Tab stop, which keeps it mounted however far it is.
    flushSync(() => {
      select?.();

      if (entry.kind === 'item') {
        const key = toGalleryItemKey(entry.item);

        setFocusedTile({ filter, index: tileKeys.indexOf(key), key });
      }
    });
    scrollToEntry(entry);
    findEntryTile(viewport, entry)?.focus({ preventScroll: true });
  };

  // A confirmation the grid's keys opened returns focus to the tile that held it, or to the Tab stop once a
  // deletion has taken that tile away.
  const getDialogReturnFocus = () => {
    const viewport = viewportRef.current;
    const opener = document.activeElement;

    if (!viewport || !(opener instanceof HTMLElement) || !viewport.contains(opener)) {
      return undefined;
    }

    return () =>
      opener.isConnected ? opener : (viewport.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR) ?? viewport);
  };

  useGalleryGridHotkeys({
    actionSelectionRefs,
    columnCount,
    getCursorCandidates,
    getDialogReturnFocus,
    getFirstVisibleTileKey,
    getFocusedItem,
    loadedItems,
    moveToEntry,
    navigationSections,
    navigateToUnloadedSlot: handleNavigateToUnloadedSlot,
    getSelectionPage,
    onNavigationStart: cancelPendingSparseNavigation,
    selectItemRange,
    toggleItem,
  });

  const settlePendingSparseNavigation = useEffectEvent(() => {
    const pending = pendingSparseNavigationRef.current;

    if (!pending) {
      return;
    }

    if (pending.filterIdentity !== sparseFilterIdentity || pending.originItemKey !== cursorKey) {
      pendingSparseNavigationRef.current = null;
      return;
    }

    if (pending.shouldRestoreFocus && pending.originFocusKey !== getFocusedTileKey()) {
      pendingSparseNavigationRef.current = null;
      return;
    }

    const item = sparseListing?.itemSlots.get(pending.index);

    if (item) {
      pendingSparseNavigationRef.current = null;
      const itemKey = toGalleryItemKey(item);
      const selectionPage = getSelectionPage(item) ?? Math.floor(pending.index / GALLERY_PAGE_SIZE);

      pending.onResolved(itemKey);

      if (pending.mode === 'select') {
        actions.selectItem(item, selectionPage);
      } else if (pending.mode === 'extend') {
        const isNavigationCurrent = () => pending.navigationGeneration === sparseNavigationGenerationRef.current;
        const isFocusCurrent = () => !pending.shouldRestoreFocus || getFocusedTileKey() === itemKey;

        void selectItemRange(item, {
          anchorKey: pending.anchorKey,
          isFocusCurrent,
          isNavigationCurrent,
          selectionPage,
        });
      }

      if (pending.shouldRestoreFocus && pending.originFocusKey) {
        pendingSparseFocusRef.current = {
          filterIdentity: pending.filterIdentity,
          itemKey,
          originFocusKey: pending.originFocusKey,
        };
        setFocusedTile({ filter, index: tileKeys.indexOf(itemKey), key: itemKey });
      }
    }
  });

  useEffect(() => {
    settlePendingSparseNavigation();
  }, [cursorKey, sparseFilterIdentity, sparseListing?.itemSlots]);

  useLayoutEffect(() => {
    const pending = pendingSparseFocusRef.current;

    if (!pending) {
      return;
    }

    if (pending.filterIdentity !== sparseFilterIdentity) {
      pendingSparseFocusRef.current = null;
      return;
    }

    if (focusedTile?.key !== pending.itemKey) {
      return;
    }

    const viewport = viewportRef.current;

    if (!viewport?.contains(document.activeElement) || getFocusedTileKey() !== pending.originFocusKey) {
      pendingSparseFocusRef.current = null;
      return;
    }

    const target = viewport.querySelector<HTMLElement>(`[data-gallery-item-key="${CSS.escape(pending.itemKey)}"]`);

    if (target) {
      pendingSparseFocusRef.current = null;
      target.focus({ preventScroll: true });
    }
  }, [focusedTile, sparseFilterIdentity, sparseListing?.itemSlots]);

  // Only explicit reveals scroll. Retry while the item loads; retire the request when another selection supersedes
  // it.
  const revealRequest = useSyncExternalStore(subscribeGalleryRevealRequests, getGalleryRevealRequest);
  const pendingRevealRef = useRef<GalleryRevealRequest | null>(null);
  // Honor requests preceding mount; selection mismatch, rather than request age, determines staleness.
  const consumedRevealTokenRef = useRef(0);
  const settlePendingReveal = useEffectEvent(() => {
    const pending = pendingRevealRef.current;

    if (!pending) {
      return;
    }

    if (pending.accountSignal.aborted) {
      pendingRevealRef.current = null;

      return;
    }

    // Another selection retires the reveal; the persisted set catches
    // off-page selections whose visible key is null.
    if (
      (gallery.selectedItemKey !== null && gallery.selectedItemKey !== pending.itemKey) ||
      (gallery.selectedItemKeys.length > 0 && !gallery.selectedItemKeys.includes(pending.itemKey))
    ) {
      pendingRevealRef.current = null;

      return;
    }

    // Consumed only once it actually scrolled, so a reveal whose row has not
    // been built yet (a strip under a collapsed disclosure, a page still
    // loading) is honored on the next row-model change.
    if (scrollToItemKey(pending.itemKey)) {
      pendingRevealRef.current = null;

      return;
    }

    // The item may live on another paginated page: follow once per reveal, so
    // a reveal whose item never materializes cannot keep pulling the user back.
    if (
      !loadedItems.some((item) => toGalleryItemKey(item) === pending.itemKey) &&
      gallery.revealTargetPage !== null &&
      gallery.revealTargetPage !== gallery.page &&
      lastPageFollowedRevealToken !== pending.token
    ) {
      lastPageFollowedRevealToken = pending.token;
      galleryCommands.setPage(gallery.revealTargetPage);
    }
  });

  useEffect(() => {
    if (revealRequest?.accountSignal.aborted) {
      if (pendingRevealRef.current?.token === revealRequest.token) {
        pendingRevealRef.current = null;
      }
    } else if (revealRequest && revealRequest.token !== consumedRevealTokenRef.current) {
      consumedRevealTokenRef.current = revealRequest.token;
      pendingRevealRef.current = revealRequest;

      if (revealRequest.absoluteIndex !== undefined && sparseListing && !isSparsePaginated) {
        const indexedItem = sparseListing.itemSlots.get(revealRequest.absoluteIndex);

        if (!indexedItem || toGalleryItemKey(indexedItem) !== revealRequest.itemKey) {
          requestSparseAbsoluteIndex(revealRequest.absoluteIndex);
        }
      }
    }

    settlePendingReveal();
  }, [isSparsePaginated, navigationSections, requestSparseAbsoluteIndex, revealRequest, sparseListing]);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return;
    }

    const setMeasuredWidth = (width: number) => {
      if (width <= 0) {
        return;
      }

      viewportWidthCache.set(region, width);
      setViewportWidth((currentWidth) => (currentWidth === width ? currentWidth : width));
    };

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;

      if (typeof width === 'number') {
        setMeasuredWidth(width);
      }
    });

    setMeasuredWidth(viewport.clientWidth);
    observer.observe(viewport);

    return () => observer.disconnect();
  }, [isEmpty, region]);

  // Measure before paint after row-model changes: unchanged visible indices otherwise leave stale offsets despite
  // new row estimates.
  const restoreSparseScrollAnchor = useEffectEvent(() => {
    const viewport = viewportRef.current;
    const previousGeometry = lastMeasuredSparseGeometryRef.current;
    const geometryChanged =
      previousGeometry.columnCount !== columnCount ||
      previousGeometry.leadingRecentRows !== leadingRecentRows ||
      previousGeometry.pinnedHeight !== pinnedHeight ||
      previousGeometry.rowHeightPx !== rowHeightPx;

    measureVirtualizer();

    if (usesSparseListing && sparseListing) {
      const pageOffset = isSparsePaginated ? sparsePageOffset : 0;
      const itemKeys = new Map(
        [...sparseListing.itemSlots.entries()].map(([index, item]) => [index, toGalleryItemKey(item)])
      );
      const currentSnapshot = {
        filterIdentity: sparseFilterIdentity,
        itemKeys,
        pageOffset,
        total: sparseListing.total,
      };
      const previousSnapshot = sparsePositionSnapshotRef.current;
      const listingIdentityChanged =
        previousSnapshot !== null &&
        (previousSnapshot.filterIdentity !== currentSnapshot.filterIdentity ||
          previousSnapshot.pageOffset !== currentSnapshot.pageOffset);
      const positionsChanged =
        previousSnapshot !== null &&
        (previousSnapshot.total !== currentSnapshot.total ||
          previousSnapshot.itemKeys.size !== itemKeys.size ||
          [...previousSnapshot.itemKeys].some(([index, itemKey]) => itemKeys.get(index) !== itemKey));

      if (listingIdentityChanged) {
        // Search and page navigation deliberately choose a new viewport. Do not carry an anchor across them.
        sparseScrollAnchorRef.current = null;
      } else if ((geometryChanged || positionsChanged) && viewport) {
        const anchor = sparseScrollAnchorRef.current;

        if (anchor) {
          const loadedMatch = [...sparseListing.itemSlots.entries()].find(
            ([, item]) => toGalleryItemKey(item) === anchor.itemKey
          );
          const anchorPageOffset = isSparsePaginated
            ? sparsePageOffset
            : Math.floor(anchor.absoluteIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
          const anchorPageState = sparseListing.pageStates.get(anchorPageOffset);
          const nearbyMatch =
            !loadedMatch && positionsChanged && anchorPageState && !anchorPageState.error && !anchorPageState.isLoading
              ? [...sparseListing.itemSlots.entries()]
                  .map(([index, item]) => ({
                    absoluteIndex: isSparsePaginated ? sparsePageOffset + index : index,
                    item,
                    localIndex: index,
                  }))
                  .sort(
                    (left, right) =>
                      Math.abs(left.absoluteIndex - anchor.absoluteIndex) -
                        Math.abs(right.absoluteIndex - anchor.absoluteIndex) || left.absoluteIndex - right.absoluteIndex
                  )[0]
              : undefined;
          const target = loadedMatch
            ? {
                absoluteIndex: isSparsePaginated ? sparsePageOffset + loadedMatch[0] : loadedMatch[0],
                item: loadedMatch[1],
                localIndex: loadedMatch[0],
              }
            : nearbyMatch;

          // A missing anchor is meaningful only after its own page has settled successfully. A page that left the
          // active sparse range, is still loading, or failed a refetch may simply have been temporarily evicted.
          if (target) {
            const rowIndex = leadingRecentRows + Math.floor(target.localIndex / columnCount);
            const rowTop = pinnedHeight + rowIndex * rowHeightPx;
            const nextScrollTop = Math.max(0, rowTop - anchor.viewportOffsetPx);

            if (target.absoluteIndex !== anchor.absoluteIndex || geometryChanged) {
              viewport.scrollTop = nextScrollTop;
            }
            sparseScrollAnchorRef.current = {
              absoluteIndex: target.absoluteIndex,
              itemKey: toGalleryItemKey(target.item),
              viewportOffsetPx: rowTop - viewport.scrollTop,
            };
          }
        }
      }

      sparsePositionSnapshotRef.current = currentSnapshot;
    } else {
      sparsePositionSnapshotRef.current = null;
    }

    lastMeasuredSparseGeometryRef.current = { columnCount, leadingRecentRows, pinnedHeight, rowHeightPx };
  });

  useLayoutEffect(() => {
    restoreSparseScrollAnchor();
  }, [
    columnCount,
    isSparsePaginated,
    leadingRecentRows,
    pinnedHeight,
    rowHeightPx,
    rows,
    sparseFilterIdentity,
    sparseListing?.itemSlots,
    sparseListing?.pageStates,
    sparseListing?.total,
    sparsePageOffset,
    usesSparseListing,
  ]);

  const virtualRows = virtualizer.virtualItems;
  const sparsePageStatusOffsets = useMemo(() => {
    if (!sparseListing) {
      return new Map<number, number>();
    }

    const scrollOffset = virtualizer.scrollOffset;
    const scrollHeight = virtualizer.scrollRect?.height;
    const visibleRows =
      scrollOffset === null || scrollHeight === undefined
        ? virtualRows
        : virtualRows.filter((row) => row.start >= scrollOffset && row.end <= scrollOffset + scrollHeight);
    const firstVisibleRow = visibleRows[0]?.index;
    const lastVisibleRow = visibleRows.at(-1)?.index;

    if (firstVisibleRow === undefined || lastVisibleRow === undefined) {
      return new Map<number, number>();
    }

    const firstBackendRow = Math.max(0, firstVisibleRow - leadingRecentRows);
    const afterLastBackendRow = Math.min(backendRowCount, lastVisibleRow + 1 - leadingRecentRows);
    const visibleStart = (isSparsePaginated ? sparsePageOffset : 0) + firstBackendRow * columnCount;
    const visibleEnd = Math.min(
      (isSparsePaginated ? sparsePageOffset : 0) + sparseBackendItemCount,
      (isSparsePaginated ? sparsePageOffset : 0) + afterLastBackendRow * columnCount
    );
    const offsets = new Map<number, number>();

    for (const [pageOffset, pageState] of sparseListing.pageStates) {
      if (!pageState.error && !pageState.isLoading) {
        continue;
      }

      const pageStart = Math.max(pageOffset, visibleStart);
      const pageEnd = Math.min(pageOffset + GALLERY_PAGE_SIZE, visibleEnd);

      for (let itemIndex = pageStart; itemIndex < pageEnd; itemIndex += 1) {
        const listingIndex = isSparsePaginated ? itemIndex - sparsePageOffset : itemIndex;

        if (!sparseListing.itemSlots.has(listingIndex)) {
          offsets.set(pageOffset, itemIndex);
          break;
        }
      }
    }

    return offsets;
  }, [
    backendRowCount,
    columnCount,
    isSparsePaginated,
    leadingRecentRows,
    sparseBackendItemCount,
    sparseListing,
    sparsePageOffset,
    virtualRows,
    virtualizer.scrollOffset,
    virtualizer.scrollRect?.height,
  ]);
  const sparsePageErrors = useMemo(
    () =>
      [...(sparseListing?.pageStates ?? [])].flatMap(([pageOffset, pageState]) =>
        pageState.error &&
        !(pageOffset === 0 && listing.status === 'stale-error') &&
        !sparsePageStatusOffsets.has(pageOffset)
          ? [{ pageOffset, pageState }]
          : []
      ),
    [listing.status, sparseListing?.pageStates, sparsePageStatusOffsets]
  );
  // From the rows in view and their overscan, never the Tab stop kept mounted far below them.
  const lastVisibleRowIndex = Math.min((virtualizer.range?.endIndex ?? 0) + GRID_OVERSCAN_ROWS, rowCount - 1);

  useEffect(() => {
    if (!usesSparseListing && paginationMode === 'infinite' && rowCount > 0 && lastVisibleRowIndex >= rowCount - 2) {
      actions.loadMore();
    }
  }, [actions, lastVisibleRowIndex, paginationMode, rowCount, usesSparseListing]);

  const handleDragEnter = useCallback((event: DragEvent) => {
    if (!dragEventContainsFiles(event)) {
      return;
    }

    event.preventDefault();
    dragDepthRef.current += 1;
    setIsDropActive(true);
  }, []);

  const handleDragLeave = useCallback((event: DragEvent) => {
    if (!dragEventContainsFiles(event)) {
      return;
    }

    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

    if (dragDepthRef.current === 0) {
      setIsDropActive(false);
    }
  }, []);

  const handleDragOver = useCallback((event: DragEvent) => {
    if (dragEventContainsFiles(event)) {
      event.preventDefault();
    }
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      dragDepthRef.current = 0;
      setIsDropActive(false);

      const files = Array.from(event.dataTransfer.files);

      if (files.length > 0) {
        void actions.uploadFiles(files);
      }
    },
    [actions]
  );

  const { inputProps: uploadInputProps, openPicker: openUploadPicker } = useGalleryUploadInput(actions.uploadFiles);

  const handleUploadKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openUploadPicker();
      }
    },
    [openUploadPicker]
  );

  const handleToggleStarredSection = useCallback(
    () => actions.updateSettings({ starredSectionCollapsed: isStarredOpen }),
    [actions, isStarredOpen]
  );

  // Show all removes the header it lives in; focus moves first, to the
  // toolbar control that reports (and undoes) the filter, so keyboard users
  // are not dropped on the document body.
  const handleShowAllStarred = useCallback(() => {
    viewportRef.current
      ?.closest('[role="tabpanel"]')
      ?.parentElement?.querySelector<HTMLElement>('[data-gallery-starred-filter-toggle]')
      ?.focus();
    actions.setStarredOnly(true);
  }, [actions]);

  const handleToggleStarred = useCallback(
    (item: GalleryItem) => void itemActions.setItemsStarred([{ kind: item.kind, name: item.name }], !item.starred),
    [itemActions]
  );
  const handleOpenItem = useCallback((item: GalleryItem) => itemActions.openItemInPreview(item), [itemActions]);

  // Releasing the anchor puts the window back over the top of the listing.
  const handleReturnToBoardTop = useCallback(() => galleryCommands.setPage(0), [galleryCommands]);

  // A successful retry unmounts the Retry that held focus; content the user can already see takes it instead,
  // without scrolling. After more items load at the end, that is the bottom of the visible grid, beside the new items.
  const focusGridContent = useCallback(() => focusVisibleGridContent(viewportRef.current, 'first'), []);
  const focusGridEnd = useCallback(() => focusVisibleGridContent(viewportRef.current, 'last'), []);

  const renderCell = useCallback(
    (item: GalleryItem, slotKey: string = toGalleryItemKey(item), selectionPage?: number) => {
      const itemKey = toGalleryItemKey(item);

      return (
        <GalleryThumbnailCell
          key={slotKey}
          alwaysShowDimensions={showImageDimensions}
          dragScope={region}
          compareRole={
            isComparisonActive && itemKey === gallery.selectedItemKey
              ? t('widgets.preview.viewing')
              : isComparisonActive && itemKey === gallery.compareImageKey
                ? t('widgets.preview.compare')
                : null
          }
          fit={thumbnailFit}
          getDragItems={getDragItems}
          getItemLabel={getReadyItemLabel}
          isPrimary={!isFollowingLive && itemKey === gallery.selectedItemKey}
          isSelected={!isFollowingLive && selectedItemKeys.has(itemKey)}
          isTabStop={itemKey === tabStopKey}
          item={item}
          onClick={handleThumbnailClick}
          onContextMenu={handleThumbnailContextMenu}
          onFocusLost={restoreTileFocus}
          onOpen={handleOpenItem}
          onToggleStarred={handleToggleStarred}
          selectionPage={selectionPage}
        />
      );
    },
    [
      gallery.compareImageKey,
      gallery.selectedItemKey,
      getDragItems,
      getReadyItemLabel,
      handleOpenItem,
      handleThumbnailClick,
      handleThumbnailContextMenu,
      handleToggleStarred,
      isComparisonActive,
      isFollowingLive,
      region,
      restoreTileFocus,
      selectedItemKeys,
      showImageDimensions,
      t,
      tabStopKey,
      thumbnailFit,
    ]
  );

  const anchoredWindowFirstItem = gallery.anchoredWindowPage * GALLERY_PAGE_SIZE + 1;
  const sparseErrorPageOffset = [...(sparseListing?.pageStates ?? [])].find(([, pageState]) => pageState.error)?.[0];
  return (
    <Stack flex="1" gap="0" h="full" minH="0" minW="0" w="full">
      {listing.status === 'stale-error' &&
      (!usesSparseListing || (Boolean(sparseListing?.pageStates.get(0)?.error) && !sparsePageStatusOffsets.has(0))) ? (
        <GalleryLoadNotice
          bg="bg.panel"
          data-gallery-page-error={usesSparseListing ? 0 : undefined}
          flexShrink={0}
          message={t('widgets.gallery.listingRefreshFailed')}
          px="2"
          py="1"
          read={listing}
          retryLabel={t('widgets.gallery.retryLoadingItems')}
          onFocusLost={focusGridContent}
        />
      ) : null}
      {usesSparseListing && listing.status !== 'error' && sparsePageErrors.length > 0 ? (
        <Stack borderBottomWidth="1px" borderColor="border.subtle" flexShrink="0" gap="1" px="2" py="1">
          {sparsePageErrors.map(({ pageOffset, pageState }) => (
            <Box key={pageOffset} data-gallery-page-error={pageOffset}>
              <GalleryPageError
                onFocusLost={pageOffset > 0 ? focusGridEnd : focusGridContent}
                pageOffset={pageOffset}
                pageState={pageState}
              />
            </Box>
          ))}
        </Stack>
      ) : null}
      <Box
        ref={syncRangeInteractionContext}
        flex="1"
        h="full"
        maxW="full"
        minH="0"
        minW="0"
        position="relative"
        w="full"
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {gallery.anchoredWindowPage > 0 && !usesSparseListing ? (
          <Flex align="center" bg="bg.panel" gap="2" justify="space-between" px="2" py="1">
            <Text color="fg.muted" fontSize="xs" truncate>
              {t('widgets.gallery.windowAnchored', { index: anchoredWindowFirstItem })}
            </Text>
            <Button flexShrink={0} size="sm" variant="ghost" onClick={handleReturnToBoardTop}>
              {t('widgets.gallery.backToBoardTop')}
            </Button>
          </Flex>
        ) : null}
        <ScrollArea.Root h="full" minH="0" variant="hover" w="full">
          {/* Focusable from script only: it holds keyboard focus while the grid has no tiles to give it. */}
          <ScrollArea.Viewport
            ref={viewportRef}
            aria-label={t('widgets.gallery.contentAriaLabel')}
            data-dnd-auto-scroll="false"
            focusVisibleRing="inside"
            h="full"
            outline="none"
            role="group"
            tabIndex={-1}
            w="full"
            onFocus={handleGridFocus}
          >
            {/* The viewport is a flex column, and `minH` replaces the content's automatic minimum: without
                `flexShrink={0}` the content would be squeezed to the viewport's height, shrinking the virtual rows'
                sizer (it holds nothing in flow) so the rows overflow it and whatever follows sits mid-list. */}
            <ScrollArea.Content display="flex" flexDirection="column" flexShrink={0} minH="full">
              {pinnedHeight > 0 ? (
                <Box
                  borderBottomWidth="1px"
                  borderColor="border.subtle"
                  data-gallery-pinned
                  flexShrink={0}
                  mb={`${GALLERY_PINNED_FOOTER_PX - 1}px`}
                  minW="0"
                >
                  {starredCells.length > 0 ? (
                    <GalleryStarredSection
                      cells={starredCells}
                      cellSizePx={cellSizePx}
                      columnCount={columnCount}
                      isOpen={isStarredOpen}
                      renderCell={renderCell}
                      state={starredStrip.state}
                      total={starredStrip.total}
                      onFocusLost={focusGridContent}
                      onShowAll={handleShowAllStarred}
                      onToggle={handleToggleStarredSection}
                    />
                  ) : isStarredFailedEmpty ? (
                    // Never an absent strip: a failed one says so where it would sit, without holding up the grid.
                    <GalleryLoadNotice
                      h={`${GALLERY_STARRED_HEADER_HEIGHT_PX}px`}
                      message={t(
                        starredStrip.state.status === 'error'
                          ? 'widgets.gallery.starredLoadFailed'
                          : 'widgets.gallery.starredRefreshFailed'
                      )}
                      px="1"
                      read={starredStrip.state}
                      retryLabel={t(
                        starredStrip.state.status === 'error'
                          ? 'widgets.gallery.retryLoadingStarredItems'
                          : 'widgets.gallery.retryRefreshingStarredItems'
                      )}
                      onFocusLost={focusGridContent}
                    />
                  ) : null}
                  <GalleryProgressSection
                    getScrollElement={getScrollElement}
                    layout={progressLayout}
                    offsetTopPx={starredLayout.height}
                  />
                </Box>
              ) : null}
              {listing.status === 'error' ? (
                <Flex
                  data-gallery-page-error={usesSparseListing ? sparseErrorPageOffset : undefined}
                  flex="1"
                  minH="8rem"
                >
                  <GalleryLoadErrorState
                    read={listing}
                    retryLabel={t('widgets.gallery.retryLoadingItems')}
                    title={t('widgets.gallery.listingLoadFailed')}
                    onFocusLost={focusGridContent}
                  />
                </Flex>
              ) : isEmpty && isStarredFailedEmpty && listing.status !== 'loading' ? (
                <Box flex="1" minH="8rem" />
              ) : isEmpty ? (
                listing.status === 'loading' || hasActiveSearch || isVirtualBoard || gallery.starredOnly ? (
                  <Flex align="center" color="fg.muted" flex="1" justify="center" minH="8rem">
                    <Text>
                      {listing.status === 'loading'
                        ? t('widgets.gallery.loadingBackendGallery')
                        : gallery.starredOnly && gallery.semanticImageQuery === null
                          ? t('widgets.gallery.noStarredItemsMatch')
                          : t('widgets.gallery.noImagesMatch')}
                    </Text>
                  </Flex>
                ) : (
                  // No inset: the zone shares the thumbnails' outer edges.
                  <Flex align="stretch" flex="1" minH="8rem">
                    <input {...uploadInputProps} />
                    <DropZone
                      alignItems="center"
                      display="flex"
                      flex="1"
                      isOver={isDropActive}
                      justifyContent="center"
                      role="button"
                      tabIndex={0}
                      onClick={openUploadPicker}
                      onKeyDown={handleUploadKeyDown}
                    >
                      <Stack align="center" gap="1">
                        <Icon as={UploadIcon} boxSize="4" color="fg.subtle" />
                        <Text color="fg.muted">{t('widgets.gallery.emptyBoardUploadHint')}</Text>
                      </Stack>
                    </DropZone>
                  </Flex>
                )
              ) : (
                <>
                  {/* Rows have a fixed pitch, so the row model sizes the grid in the same render. The virtualizer's
                      total trails a row-count change by a commit, long enough for the notice after the rows to
                      leave first and the browser to clamp a scroll position resting at the bottom. */}
                  <Box h={`${rowCount * rowHeightPx}px`} position="relative" w="full">
                    <Box
                      aria-label={t('widgets.gallery.itemsAriaLabel')}
                      h="full"
                      inset="0"
                      position="absolute"
                      role="list"
                      w="full"
                    >
                      {virtualRows.map((virtualRow) => {
                        if (usesSparseListing && sparseListing) {
                          const recentRow = sparseRecentAtTop ? virtualRow.index : virtualRow.index - backendRowCount;
                          const isRecentRow = sparseRecentAtTop
                            ? virtualRow.index < sparseRecentRowCount
                            : virtualRow.index >= backendRowCount;

                          if (isRecentRow) {
                            const firstRecentIndex = recentRow * columnCount;
                            const recentRowItems = sparseRecentItems.slice(
                              firstRecentIndex,
                              firstRecentIndex + columnCount
                            );

                            return (
                              <Box
                                key={virtualRow.key}
                                data-gallery-section="recent"
                                display="grid"
                                gap={`${GALLERY_GRID_GAP_PX}px`}
                                gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
                                h={`${cellSizePx}px`}
                                left="0"
                                position="absolute"
                                role="presentation"
                                top="0"
                                transform={`translateY(${virtualRow.start - pinnedHeight}px)`}
                                w="full"
                              >
                                {recentRowItems.map((item) =>
                                  renderCell(item, `recent-slot:${toGalleryItemKey(item)}`)
                                )}
                              </Box>
                            );
                          }

                          const backendRow = virtualRow.index - leadingRecentRows;
                          const firstItemIndex = backendRow * columnCount;
                          const itemCount = Math.max(0, Math.min(columnCount, sparseBackendItemCount - firstItemIndex));
                          const cells = Array.from({ length: itemCount }, (_, column) => {
                            const itemIndex = firstItemIndex + column;
                            const item = sparseListing.itemSlots.get(itemIndex);
                            const absoluteItemIndex = sparsePageOffset + itemIndex;

                            if (item) {
                              return renderCell(
                                item,
                                getGallerySparseSlotKey(absoluteItemIndex),
                                Math.floor(absoluteItemIndex / GALLERY_PAGE_SIZE)
                              );
                            }

                            const pageOffset = Math.floor(absoluteItemIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;

                            return (
                              <GallerySparseSlot
                                key={getGallerySparseSlotKey(absoluteItemIndex)}
                                cellSizePx={cellSizePx}
                                onFocusLost={focusGridEnd}
                                pageOffset={pageOffset}
                                pageState={sparseListing.pageStates.get(pageOffset)}
                                showPageStatus={sparsePageStatusOffsets.get(pageOffset) === absoluteItemIndex}
                              />
                            );
                          });

                          return (
                            <Box
                              key={virtualRow.key}
                              data-gallery-section="regular"
                              display="grid"
                              gap={`${GALLERY_GRID_GAP_PX}px`}
                              gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
                              h={`${cellSizePx}px`}
                              left="0"
                              position="absolute"
                              role="presentation"
                              top="0"
                              transform={`translateY(${virtualRow.start - pinnedHeight}px)`}
                              w="full"
                            >
                              {cells}
                            </Box>
                          );
                        }

                        const row = rows[virtualRow.index];

                        if (!row) {
                          return null;
                        }

                        return (
                          <Box
                            key={virtualRow.key}
                            data-gallery-section={row.section}
                            display="grid"
                            gap={`${GALLERY_GRID_GAP_PX}px`}
                            gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
                            left="0"
                            position="absolute"
                            role="presentation"
                            top="0"
                            transform={`translateY(${virtualRow.start - pinnedHeight}px)`}
                            w="full"
                          >
                            {row.cells.map((item) => renderCell(item))}
                          </Box>
                        );
                      })}
                    </Box>
                  </Box>
                  {!usesSparseListing && listing.status === 'more-error' ? (
                    // Loaded pages stay; only the page that failed waits on Retry.
                    <GalleryLoadNotice
                      justify="center"
                      message={t('widgets.gallery.listingLoadMoreFailed')}
                      px="2"
                      py="2"
                      read={listing}
                      retryLabel={t('widgets.gallery.retryLoadingMoreItems')}
                      onFocusLost={focusGridEnd}
                    />
                  ) : !usesSparseListing &&
                    paginationMode === 'infinite' &&
                    (listing.isFetchingMore || listing.status === 'loading') &&
                    gallery.items.length > 0 ? (
                    <Flex align="center" justify="center" py="2">
                      <Spinner color="fg.subtle" />
                    </Flex>
                  ) : null}
                  {!usesSparseListing &&
                    paginationMode === 'infinite' &&
                    !listing.isFetchingMore &&
                    isWindowTruncated && (
                      <Flex align="center" justify="center" py="3">
                        <Text color="fg.subtle" fontSize="md" textAlign="center">
                          {gallery.anchoredWindowPage > 0
                            ? t('widgets.gallery.windowLimitFrom', {
                                count: gallery.items.length,
                                index: anchoredWindowFirstItem,
                              })
                            : t('widgets.gallery.windowLimit', { count: gallery.items.length })}
                        </Text>
                      </Flex>
                    )}
                </>
              )}
            </ScrollArea.Content>
          </ScrollArea.Viewport>
          <ScrollArea.Scrollbar>
            <ScrollArea.Thumb />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
        {isDropActive && (
          <DropZone
            alignItems="center"
            display="flex"
            flexDirection="column"
            gap="2"
            inset="0"
            isOver
            justifyContent="center"
            pointerEvents="none"
            position="absolute"
            variant="overlay"
            zIndex="1"
          >
            <UploadIcon size="20" />
            <Text fontSize="md" fontWeight="600">
              {t('widgets.gallery.dropMediaToUploadToBoard', { name: selectedBoardName })}
            </Text>
          </DropZone>
        )}
        <ImageContextMenu boards={gallery.boards} target={activeContextMenuTarget} onClose={handleCloseContextMenu} />
      </Box>
    </Stack>
  );
};
