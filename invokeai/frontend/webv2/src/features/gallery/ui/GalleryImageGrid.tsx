import { Box, chakra, Flex, HStack, Icon, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey, type GalleryItem, type GalleryItemKey } from '@features/gallery/core/items';
import {
  getGalleryRevealRequest,
  getGallerySessionNavigationKey,
  subscribeGalleryRevealRequests,
  type GalleryNavigationEntry,
  type GalleryRevealRequest,
} from '@features/gallery/core/selection';
import { isDateBoardId } from '@features/gallery/data/backend';
import { GALLERY_PAGE_SIZE, imageIndexAvailabilityOptions } from '@features/gallery/data/queries';
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
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
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
import { GalleryProgressSection } from './GalleryProgressSection';
import { GalleryThumbnailCell } from './GalleryThumbnail';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';
import { useGalleryGridHotkeys } from './useGalleryGridHotkeys';
import { useGalleryGridSelection } from './useGalleryGridSelection';
import { useGalleryUploadInput } from './useGalleryUploadInput';

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

const GalleryPageError = ({ pageState }: { pageState: GallerySparsePageState }) => {
  const { t } = useTranslation();
  const handleRetry = useCallback(() => void pageState.retry(), [pageState]);

  return (
    <Stack align="center" gap="1" maxW="full" px="1">
      <Text color="fg.muted" fontSize="xs" lineClamp={2} textAlign="center">
        {pageState.error?.message}
      </Text>
      <Button color="fg" size="sm" variant="ghost" onClick={handleRetry}>
        {t('common.retry')}
      </Button>
    </Stack>
  );
};

const GallerySparseSlot = ({
  cellSizePx,
  pageState,
  showPageStatus,
}: {
  cellSizePx: number;
  pageState: GallerySparsePageState | undefined;
  showPageStatus: boolean;
}) => {
  const { t } = useTranslation();
  const error = pageState?.error;
  const state = error && showPageStatus ? 'error' : pageState?.isLoading && showPageStatus ? 'loading' : 'empty';
  const retry = useCallback(() => void pageState?.retry(), [pageState]);

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
    >
      {state === 'loading' ? (
        <Stack align="center" aria-label={t('widgets.gallery.loadingBackendGallery')} gap="1" role="status">
          <Spinner aria-hidden="true" size="md" />
          <Text color="fg.muted" fontSize="xs" lineClamp={1}>
            {t('widgets.gallery.loadingBackendGallery')}
          </Text>
        </Stack>
      ) : state === 'error' && pageState ? (
        <Stack align="center" gap="0" maxW="full" px="1" role="group">
          <Text aria-live="polite" color="fg.muted" fontSize="xs" lineClamp={1} textAlign="center">
            {error?.message}
          </Text>
          <Button color="fg" size="sm" variant="ghost" onClick={retry}>
            {t('common.retry')}
          </Button>
        </Stack>
      ) : null}
    </Box>
  );
};

/** Show all appears only when starred items exceed the strip and activates the starred-only listing. */
const GalleryStarredSectionHeader = ({
  isOpen,
  onShowAll,
  onToggle,
  shownCount,
  total,
}: {
  isOpen: boolean;
  onShowAll: () => void;
  onToggle: () => void;
  shownCount: number;
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
  total,
  onShowAll,
  onToggle,
}: {
  cells: GalleryItem[];
  /** Rows take the listing's pitch explicitly, so the pinned block measures what the layout computed. */
  cellSizePx: number;
  columnCount: number;
  isOpen: boolean;
  renderCell: (item: GalleryItem) => ReactNode;
  total: number;
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
        total={total}
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

/** Measure viewport width for columns so both layouts share the same grid. */
export const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const {
    actions,
    filter,
    gallery,
    isWindowTruncated,
    itemActions,
    region,
    setVisibleRange,
    sparseListing,
    starredStrip,
  } = useGalleryWidget();
  const {
    gallery: galleryCommands,
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

  const {
    actionSelectionRefs,
    activeContextMenuTarget,
    getDragItems,
    handleCloseContextMenu,
    handleThumbnailClick,
    handleThumbnailContextMenu,
    loadedItems,
    selectedItemKeys,
    syncRangeInteractionContext,
  } = useGalleryGridSelection();

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
  const sparsePageStatusOffsets = useMemo(() => {
    if (!sparseListing) {
      return new Map<number, number>();
    }

    const offsets = new Map<number, number>();
    for (const [pageOffset, pageState] of sparseListing.pageStates) {
      if (!pageState.error && !pageState.isLoading) {
        continue;
      }

      const visibleRangeStart = isSparsePaginated ? sparsePageOffset : 0;
      const visibleRangeEnd = visibleRangeStart + sparseBackendItemCount;
      const pageStart = Math.max(pageOffset, visibleRangeStart);
      const pageEnd = Math.min(pageOffset + GALLERY_PAGE_SIZE, visibleRangeEnd);
      for (let itemIndex = pageStart; itemIndex < pageEnd; itemIndex += 1) {
        const listingIndex = isSparsePaginated ? itemIndex - sparsePageOffset : itemIndex;
        if (!sparseListing.itemSlots.has(listingIndex)) {
          offsets.set(pageOffset, itemIndex);
          break;
        }
      }
    }
    return offsets;
  }, [isSparsePaginated, sparseBackendItemCount, sparseListing, sparsePageOffset]);
  const sparsePageErrors = useMemo(
    () =>
      [...(sparseListing?.pageStates ?? [])].flatMap(([pageOffset, pageState]) =>
        pageState.error && !sparsePageStatusOffsets.has(pageOffset) ? [{ pageOffset, pageState }] : []
      ),
    [sparseListing?.pageStates, sparsePageStatusOffsets]
  );
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
  const navigationSections = useMemo((): GalleryNavigationEntry[][] => {
    const shownStripItems = isStarredOpen ? starredCells : [];
    const selectedKey = gallery.selectedItemKey;
    const isSelected = (item: GalleryItem) => toGalleryItemKey(item) === selectedKey;
    const hiddenStripSelection =
      selectedKey !== null && !shownStripItems.some(isSelected) && !gallery.items.some(isSelected)
        ? starredStrip.items.find(isSelected)
        : undefined;
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
    gallery.selectedItemKey,
    columnCount,
    isProgressOpen,
    isStarredOpen,
    progressSessions,
    sparseListing,
    isSparsePaginated,
    sparseBackendItemCount,
    sparseRecentAtTop,
    sparseRecentItems,
    starredCells,
    starredStrip.items,
    usesSparseListing,
  ]);
  const cursorKey =
    followedProgressSessionId !== null
      ? getGallerySessionNavigationKey(followedProgressSessionId)
      : gallery.selectedItemKey;
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
    (instance: { getVirtualItems: () => readonly { index: number }[] }) => {
      if (!usesSparseListing || !sparseListing) {
        return;
      }

      const visibleItems = instance.getVirtualItems();
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
    overscan: 4,
  });
  const pendingSparseNavigationRef = useRef<{
    filterIdentity: string;
    index: number;
    originItemKey: string | null;
  } | null>(null);
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
    (index: number) => {
      pendingSparseNavigationRef.current = {
        filterIdentity: sparseFilterIdentity,
        index,
        originItemKey: cursorKey,
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
    } else {
      scrollToItemKey(toGalleryItemKey(entry.item));
    }
  };

  useGalleryGridHotkeys({
    actionSelectionRefs,
    columnCount,
    cursorKey,
    loadedItems,
    navigationSections,
    navigateToUnloadedSlot: handleNavigateToUnloadedSlot,
    getSelectionPage,
    scrollToEntry,
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

    const item = sparseListing?.itemSlots.get(pending.index);

    if (item) {
      pendingSparseNavigationRef.current = null;
      actions.selectItem(item, Math.floor(pending.index / GALLERY_PAGE_SIZE));
    }
  });

  useEffect(() => {
    settlePendingSparseNavigation();
  }, [cursorKey, sparseFilterIdentity, sparseListing?.itemSlots]);

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
  const lastVisibleRowIndex = virtualRows[virtualRows.length - 1]?.index ?? 0;

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

  // Releasing the anchor puts the window back over the top of the listing.
  const handleReturnToBoardTop = useCallback(() => galleryCommands.setPage(0), [galleryCommands]);

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
          item={item}
          onClick={handleThumbnailClick}
          onContextMenu={handleThumbnailContextMenu}
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
      handleThumbnailClick,
      handleThumbnailContextMenu,
      handleToggleStarred,
      isComparisonActive,
      isFollowingLive,
      region,
      selectedItemKeys,
      showImageDimensions,
      t,
      thumbnailFit,
    ]
  );

  const anchoredWindowFirstItem = gallery.anchoredWindowPage * GALLERY_PAGE_SIZE + 1;
  const initialSparsePageState = sparseListing?.pageStates.get(isSparsePaginated ? sparsePageOffset : 0);

  return (
    <Stack flex="1" gap="0" h="full" minH="0" minW="0" w="full">
      {sparsePageErrors.length > 0 ? (
        <Stack borderBottomWidth="1px" borderColor="border.subtle" flexShrink="0" gap="1" px="2" py="1">
          {sparsePageErrors.map(({ pageOffset, pageState }) => (
            <Box key={pageOffset} data-gallery-page-error={pageOffset}>
              <GalleryPageError pageState={pageState} />
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
          <ScrollArea.Viewport ref={viewportRef} data-dnd-auto-scroll="false" h="full" outline="none" w="full">
            <ScrollArea.Content display="flex" flexDirection="column" minH="full">
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
                      total={starredStrip.total}
                      onShowAll={handleShowAllStarred}
                      onToggle={handleToggleStarredSection}
                    />
                  ) : null}
                  <GalleryProgressSection
                    getScrollElement={getScrollElement}
                    layout={progressLayout}
                    offsetTopPx={starredLayout.height}
                  />
                </Box>
              ) : null}
              {isEmpty ? (
                gallery.isLoading ||
                hasActiveSearch ||
                isVirtualBoard ||
                gallery.starredOnly ||
                initialSparsePageState?.error ? (
                  <Flex align="center" color="fg.muted" flex="1" justify="center" minH="8rem">
                    {initialSparsePageState?.error ? null : (
                      <Text>
                        {gallery.isLoading
                          ? t('widgets.gallery.loadingBackendGallery')
                          : gallery.starredOnly && gallery.semanticImageQuery === null
                            ? t('widgets.gallery.noStarredItemsMatch')
                            : t('widgets.gallery.noImagesMatch')}
                      </Text>
                    )}
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
                  <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
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
                  {!usesSparseListing &&
                    paginationMode === 'infinite' &&
                    gallery.isLoading &&
                    gallery.items.length > 0 && (
                      <Flex align="center" justify="center" py="2">
                        <Spinner color="fg.subtle" />
                      </Flex>
                    )}
                  {!usesSparseListing && paginationMode === 'infinite' && !gallery.isLoading && isWindowTruncated && (
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
