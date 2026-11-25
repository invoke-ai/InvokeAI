import { Box, chakra, Flex, HStack, Icon, ScrollArea, Spinner, Stack, Text, VisuallyHidden } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { getGalleryCullMode, getGalleryCullModeValues } from '@features/gallery/core/cull';
import {
  toGalleryItemKey,
  type GalleryItem,
  type GalleryItemKey,
  type GalleryItemRef,
} from '@features/gallery/core/items';
import {
  EMPTY_GALLERY_MARKS_FILTER,
  getGalleryItemMarks,
  isGalleryMarksFilterActive,
  toggleGalleryBookmark,
  type GalleryMarksPatch,
} from '@features/gallery/core/marks';
import {
  getGalleryRevealRequest,
  getGallerySessionNavigationKey,
  getSelectedGalleryItemFromValues,
  subscribeGalleryRevealRequests,
  type GalleryNavigationEntry,
  type GalleryRevealRequest,
} from '@features/gallery/core/selection';
import { gallerySemanticReferenceKey } from '@features/gallery/core/semanticImageQuery';
import { isDateBoardId } from '@features/gallery/data/backend';
import {
  GALLERY_PAGE_SIZE,
  imageIndexAvailabilityOptions,
  type GalleryItemsFilter,
} from '@features/gallery/data/queries';
import { isEventFromOwnDom } from '@platform/react/isEventFromOwnDom';
import { useMountEffect } from '@platform/react/useMountEffect';
import { getApiErrorMessage } from '@platform/transport/http';
import { Button, DropZone } from '@platform/ui';
import { LoadError, RetryButton } from '@platform/ui/LoadError';
import { useQuery } from '@tanstack/react-query';
import { BookmarkIcon, ChevronRightIcon, UploadIcon } from 'lucide-react';
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
  type RefObject,
} from 'react';
import { flushSync } from 'react-dom';
import { defaultRangeExtractor, useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GallerySparsePageState } from './useGalleryData';

import { GalleryCullCommands, type GalleryCullTarget } from './GalleryCullCommands';
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
import { focusVisibleOperable, GalleryLoadNotice } from './GalleryLoadError';
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

/** Seeded per region so a remount does not paint fallback-sized tiles before the ResizeObserver reports. */
const viewportWidthCache = new Map<string, number>();
const STARRED_TRIGGER_HOVER_STYLES = { color: 'fg' } as const;
const EMPTY_GALLERY_ITEMS: GalleryItem[] = [];

// Module-scoped so a grid remount cannot replay an already-followed reveal.
let lastPageFollowedRevealToken = 0;

// React bubbles drags over portalled content (a tile's context menu); those are not over the grid.
const isFileDragOnGrid = (event: DragEvent): boolean =>
  isEventFromOwnDom(event) && Array.from(event.dataTransfer.types).includes('Files');

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
        <RetryButton
          aria-label={t(isMoreError ? 'widgets.gallery.retryLoadingMoreItems' : 'widgets.gallery.retryLoadingItems')}
          color="fg"
          flagsFailure
          isRetrying={read.isRetrying}
          onRetry={read.retry}
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
      <RetryButton
        aria-label={t(isMoreError ? 'widgets.gallery.retryLoadingMoreItems' : 'widgets.gallery.retryLoadingItems')}
        color="fg"
        flagsFailure
        isRetrying={read.isRetrying}
        onRetry={read.retry}
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
      {/* aria-expanded carries the open state. */}
      <chakra.button
        aria-expanded={isOpen}
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
        <Icon as={BookmarkIcon} boxSize="3" fill="currentColor" />
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
            {t('widgets.gallery.bookmarkedItems')}
          </Text>
          <Text as="span" color="currentColor" fontSize="xs" fontVariantNumeric="tabular-nums" lineHeight="1">
            {total}
          </Text>
        </HStack>
      </chakra.button>
      {state.status === 'stale-error' ? (
        <RetryButton
          aria-label={t('widgets.gallery.retryRefreshingBookmarkedItems')}
          color="fg.error"
          flagsFailure
          isRetrying={state.isRetrying}
          onRetry={state.retry}
          size="xs"
          variant="ghost"
          onFocusLost={onFocusLost}
        />
      ) : null}
      {total > shownCount ? (
        <Button
          aria-label={t('widgets.gallery.showAllBookmarkedItems')}
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

/** Outside virtualization but in its place in cross-section keyboard navigation. */
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
        <Box aria-label={t('widgets.gallery.bookmarkedItems')} role="list">
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

const TILE_BUTTON_SELECTOR = '[role="listitem"] button[aria-pressed]';

const NO_ITEMS: GalleryItem[] = [];
const GRID_OVERSCAN_ROWS = 4;

interface FocusedTile {
  filter: GalleryItemsFilter;
  index: number;
  key: GalleryItemKey;
  /** The sparse slot whose next occupant inherits the Tab stop. */
  slot?: number;
}

const extractRangeKeeping = (range: VirtualRange, keptRow: number): number[] => {
  const indexes = defaultRangeExtractor(range);

  return keptRow < 0 || keptRow >= range.count || indexes.includes(keptRow)
    ? indexes
    : [...indexes, keptRow].sort((a, b) => a - b);
};

const getTileItemKey = (element: Element | null): GalleryItemKey | null =>
  (element
    ?.closest('[role="listitem"]')
    ?.querySelector('[data-gallery-item-key]')
    ?.getAttribute('data-gallery-item-key') ?? null) as GalleryItemKey | null;

const findEntryTile = (viewport: HTMLElement, entry: GalleryNavigationEntry): HTMLElement | null =>
  entry.kind === 'session'
    ? viewport.querySelector<HTMLElement>(`[data-gallery-session-id="${CSS.escape(entry.id)}"]`)
    : entry.kind === 'item'
      ? viewport.querySelector<HTMLElement>(`[data-gallery-item-key="${CSS.escape(toGalleryItemKey(entry.item))}"]`)
      : null;

const focusVisibleGridContent = (viewport: HTMLElement | null, edge: 'first' | 'last') => {
  if (!focusVisibleOperable(viewport, { edge, selector: TILE_BUTTON_SELECTOR })) {
    focusVisibleOperable(viewport, { edge });
  }
};

const GALLERY_EXIT_COMPARE_COMMAND = 'gallery.exitCompare';

/** Mounted only while comparing, so Esc keeps its other meanings the rest of the time. */
const GalleryExitCompareHotkey = () => {
  const { t } = useTranslation();
  const { actions, runtime } = useGalleryWidget();

  useMountEffect(() => {
    const exitCompare = () => actions.setCompareItem(null);
    const title = t('widgets.preview.commands.exitCompare');
    const disposers = [
      runtime.commands.register({ handler: exitCompare, id: GALLERY_EXIT_COMPARE_COMMAND, title }),
      runtime.hotkeys.register({
        commandId: GALLERY_EXIT_COMPARE_COMMAND,
        defaultKeys: ['esc'],
        id: GALLERY_EXIT_COMPARE_COMMAND,
        title,
      }),
    ];

    return () => disposers.forEach((dispose) => dispose());
  });

  return null;
};

export const GalleryImageGrid = ({
  mediaFilterTriggerRef,
}: {
  mediaFilterTriggerRef?: RefObject<HTMLButtonElement | null>;
} = {}) => {
  const { i18n, t } = useTranslation();
  const {
    actions,
    filter,
    gallery,
    itemActions,
    listing,
    pinRevealIndex,
    projectId,
    region,
    runtime,
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
    widgets,
  } = useGalleryUi();
  const { data: indexAvailability } = useQuery(imageIndexAvailabilityOptions());
  const getReadyItemLabel = indexAvailability?.state === 'ready' ? getItemLabel : null;
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(region) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const {
    imageDensityPercent,
    mediaFilter,
    middleClickOpensInNewTab,
    paginationMode,
    progressSectionCollapsed,
    showImageDimensions,
    showPendingItems,
    starredSectionCollapsed,
    thumbnailFit,
  } = gallery.settings;
  const isStarredOpen = !starredSectionCollapsed;
  const cullMode = getGalleryCullMode(galleryValues);
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
  const getItemReturnFocus = useCallback(
    (item: GalleryItem) => () => {
      const viewport = viewportRef.current;
      const selector = `button[data-gallery-item-key="${CSS.escape(toGalleryItemKey(item))}"]`;

      return (
        viewport?.querySelector<HTMLElement>(selector) ??
        viewport?.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR) ??
        viewport ??
        null
      );
    },
    []
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
    selectionMarks,
    selectItemRange,
    syncRangeInteractionContext,
    toggleItem,
  } = useGalleryGridSelection({ getItemReturnFocus, getSelectionPage });

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
  // Not hidden while live frames show: the waiting image stays marked.
  const isCompareArmed = gallery.isCompareArmed;
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const selectedBoardName = selectedBoard
    ? getGalleryBoardLabel(selectedBoard, t)
    : t('widgets.gallery.selectedBoardFallback');
  // Without a flag filter the listing leaves bookmarked items out, so an all-bookmarked board still has the strip.
  const isEmpty =
    (usesSparseListing ? sparseBackendItemCount === 0 && sparseRecentItems.length === 0 : gallery.items.length === 0) &&
    starredStrip.items.length === 0;
  const isStarredFailedEmpty =
    (starredStrip.state.status === 'error' || starredStrip.state.status === 'stale-error') &&
    starredStrip.items.length === 0;
  // A ranking that matched nothing is still a search result, never an empty board inviting an upload.
  const hasActiveSearch = gallery.searchTerm.trim() !== '' || gallery.semanticImageQuery !== null;
  const isVirtualBoard = isDateBoardId(gallery.selectedBoardId);

  const rows = useMemo(() => buildGalleryGridRows(gallery.items, columnCount), [columnCount, gallery.items]);
  const starredCells = useMemo(
    () => getGalleryStarredStripItems(starredStrip.items, columnCount),
    [columnCount, starredStrip.items]
  );
  const isProgressOpen = showPendingItems && !progressSectionCollapsed;
  const shownStripItems = isStarredOpen ? starredCells : NO_ITEMS;
  // A bookmarked selection no tile shows still belongs to the strip, so arrows step from it, but only when made in this
  // listing: a leftover would add a phantom entry. A flag-filtered listing, ranking or anchored window has none.
  const isStripApplicable =
    gallery.marksFilter.flags.length === 0 && gallery.semanticImageQuery === null && gallery.anchoredWindowPage === 0;
  const hasMarksFilter = gallery.semanticImageQuery === null && isGalleryMarksFilterActive(gallery.marksFilter);
  const hiddenStripSelection = useMemo((): GalleryItem | null => {
    const selectedItem = getSelectedGalleryItemFromValues(galleryValues);
    const selectedKey = gallery.selectedItemKey ?? (selectedItem ? toGalleryItemKey(selectedItem) : null);
    const isSelected = (item: GalleryItem) => toGalleryItemKey(item) === selectedKey;

    if (
      !isStripApplicable ||
      selectedKey === null ||
      shownStripItems.some(isSelected) ||
      gallery.items.some(isSelected)
    ) {
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
  }, [gallery, galleryValues, isStripApplicable, shownStripItems, starredStrip.items]);
  const navigationSections = useMemo((): GalleryNavigationEntry[][] => {
    const stripEntries: GalleryNavigationEntry[] = shownStripItems.map((item) => ({ item, kind: 'item' }));

    if (hiddenStripSelection) {
      stripEntries.push({ item: hiddenStripSelection, kind: 'item' });
    }

    const loadingPageOffsets = [...(sparseListing?.pageStates ?? [])]
      .filter(([, pageState]) => pageState.isLoading)
      .map(([offset]) => offset);
    const pendingPageOffsets = new Set(
      isSparsePaginated ? (loadingPageOffsets.length > 0 ? [0] : []) : loadingPageOffsets
    );
    const listingEntries =
      usesSparseListing && sparseListing
        ? buildSparseGalleryNavigationEntries({
            columnCount,
            includeUnloadedBoundaries: !isSparsePaginated,
            itemSlots: sparseListing.itemSlots,
            pageOffsets: isSparsePaginated ? [0] : [...sparseListing.pageStates.keys()],
            pendingPageOffsets,
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
  // The persisted primary, unlike the visible key, survives its page leaving the subscribed range.
  const navigationOriginKey =
    followedProgressSessionId !== null
      ? getGallerySessionNavigationKey(followedProgressSessionId)
      : gallery.primarySelectedItemKey;
  // Sparse slots keep their real row positions even when pages are unloaded.
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
  const focusedSlotItem = focusedTile?.slot === undefined ? undefined : sparseListing?.itemSlots.get(focusedTile.slot);
  const tabStopIndex =
    focusedTileIndex >= 0
      ? focusedTileIndex
      : selectedTileIndex >= 0
        ? selectedTileIndex
        : focusedTile?.filter === filter
          ? focusedTile.slot === undefined
            ? Math.max(0, Math.min(focusedTile.index, tileKeys.length - 1))
            : Math.max(0, focusedSlotItem ? tileKeys.indexOf(toGalleryItemKey(focusedSlotItem)) : 0)
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
  // Never unmount the Tab stop: focus would fall to the body and Tab would skip the grid.
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

      // The kept Tab stop row must not widen the sparse Query subscription or become the scroll anchor.
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
        // Descending recents precede slot zero; before count discovery ascending recents keep page zero active.
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
  /**
   * A `verified` index from a reveal may exceed a stale total, so its page is pinned. `keepIndex` keeps a neighboring
   * origin page subscribed until an arrow step lands.
   */
  const requestSparseAbsoluteIndex = useCallback(
    (absoluteIndex: number, { keepIndex, verified = false }: { keepIndex?: number; verified?: boolean } = {}) => {
      if (!sparseListing || isSparsePaginated || !setVisibleRange) {
        return;
      }

      if (absoluteIndex < 0 || (!verified && sparseListing.total !== null && absoluteIndex >= sparseListing.total)) {
        return;
      }

      if (verified) {
        pinRevealIndex?.(absoluteIndex);
      }

      const pageOffset = Math.floor(absoluteIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
      const keepPageOffset =
        keepIndex === undefined ? pageOffset : Math.floor(keepIndex / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
      // Only a neighboring origin stays: a far one would subscribe every page between the two.
      const keptPageOffset = Math.abs(keepPageOffset - pageOffset) <= GALLERY_PAGE_SIZE ? keepPageOffset : pageOffset;

      // Narrowing to an already subscribed page would drop the cursor's page.
      if (!sparseListing.pageStates.has(pageOffset)) {
        setVisibleRange({
          endIndexExclusive: Math.min(
            sparseListing.total ?? Number.POSITIVE_INFINITY,
            Math.max(pageOffset, keptPageOffset) + GALLERY_PAGE_SIZE
          ),
          startIndex: Math.min(pageOffset, keptPageOffset),
        });
      }
      virtualizer.scrollToIndex(leadingRecentRows + Math.floor(absoluteIndex / columnCount));
    },
    [columnCount, isSparsePaginated, leadingRecentRows, pinRevealIndex, setVisibleRange, sparseListing, virtualizer]
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

      const originFocusKey = shouldRestoreFocus
        ? activeSessionId
          ? getGallerySessionNavigationKey(activeSessionId)
          : getTileItemKey(active)
        : null;
      const stepOriginKey = originFocusKey ?? navigationOriginKey;
      const originSlot = sparseSlotEntries.find(([, item]) => toGalleryItemKey(item) === stepOriginKey)?.[0];

      pendingSparseFocusRef.current = null;
      pendingSparseNavigationRef.current = {
        anchorKey: request.anchorKey,
        filterIdentity: sparseFilterIdentity,
        index,
        mode: request.mode,
        navigationGeneration: sparseNavigationGenerationRef.current,
        onResolved: request.onResolved,
        originItemKey: navigationOriginKey,
        originFocusKey,
        shouldRestoreFocus,
      };
      requestSparseAbsoluteIndex(index, { keepIndex: originSlot });
    },
    [navigationOriginKey, requestSparseAbsoluteIndex, sparseFilterIdentity, sparseSlotEntries]
  );

  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  // A stable identity adds nothing and hinders compiler memoization.
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

    if (isStarredOpen && starredCells.some((item) => toGalleryItemKey(item) === itemKey)) {
      viewportRef.current?.scrollTo({ top: 0 });
      return true;
    }

    return false;
  };
  const scrollToEntry = (entry: GalleryNavigationEntry) => {
    if (entry.kind === 'session') {
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
        const slot = sparseSlotEntries.find(([, item]) => toGalleryItemKey(item) === key)?.[0];

        setFocusedTile((current) =>
          current?.key === key && current.index === index && current.slot === slot && current.filter === filter
            ? current
            : { filter, index, key: key as GalleryItemKey, slot }
        );
      }
    },
    [filter, sparseSlotEntries, tileKeys]
  );

  // A tile that leaves the document holding focus takes it along: the same item takes it back where it now shows,
  // else the Tab stop, first tile in view or the grid. A user who already went elsewhere keeps their focus. A slot's
  // tile that a shifted listing hands another item while focused loses it too: focus follows the item, not the slot.
  const restoreTileFocus = useCallback((itemKey: GalleryItemKey) => {
    const viewport = viewportRef.current;
    const active = document.activeElement;
    const activeTileKey = getTileItemKey(active);
    const isReusedTile = activeTileKey !== null && activeTileKey !== itemKey && viewport?.contains(active) === true;

    if (!viewport || (active !== null && active !== document.body && !isReusedTile)) {
      return;
    }

    const sameItem = viewport.querySelector<HTMLElement>(`[data-gallery-item-key="${CSS.escape(itemKey)}"]`);

    if (sameItem) {
      sameItem.focus({ preventScroll: true });
    } else if (
      !focusVisibleOperable(viewport, { selector: GALLERY_TAB_STOP_SELECTOR }) &&
      !focusVisibleOperable(viewport, { selector: TILE_BUTTON_SELECTOR })
    ) {
      viewport.focus({ preventScroll: true });
    }
  }, []);

  const getFocusedTileKey = (): string | null => {
    const viewport = viewportRef.current;
    const active = document.activeElement;

    if (!viewport || !(active instanceof HTMLElement) || !viewport.contains(active)) {
      return null;
    }

    const sessionId = active.closest('[data-gallery-session-id]')?.getAttribute('data-gallery-session-id');

    return sessionId ? getGallerySessionNavigationKey(sessionId) : getTileItemKey(active);
  };
  // A followed session the grid does not show gives way to the selection.
  const getCursorCandidates = () => [
    getFocusedTileKey(),
    followedProgressSessionId === null ? null : getGallerySessionNavigationKey(followedProgressSessionId),
    hiddenStripSelection ? toGalleryItemKey(hiddenStripSelection) : gallery.selectedItemKey,
  ];
  const getFirstVisibleTileKey = (): string | null => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return null;
    }

    const bounds = viewport.getBoundingClientRect();

    // Document order is visual order.
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

  /** Moves focus only while it is in the grid; a command run elsewhere leaves focus be. */
  const moveToEntry = (entry: GalleryNavigationEntry, select: (() => void) | null) => {
    const viewport = viewportRef.current;

    if (!viewport?.contains(document.activeElement)) {
      if (select) {
        select();
        scrollToEntry(entry);
      }
      return;
    }

    // Committed before focus moves: the Tab stop stays mounted however far it is.
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

  const getDialogReturnFocus = () => {
    const viewport = viewportRef.current;
    const opener = document.activeElement;

    if (!viewport || !(opener instanceof HTMLElement) || !viewport.contains(opener)) {
      return undefined;
    }

    return () =>
      opener.isConnected ? opener : (viewport.querySelector<HTMLElement>(GALLERY_TAB_STOP_SELECTOR) ?? viewport);
  };

  const { cullStepper } = useGalleryGridHotkeys({
    actionSelectionRefs,
    columnCount,
    getCursorCandidates,
    getDialogReturnFocus,
    getFirstVisibleTileKey,
    getFocusedItem,
    moveToEntry,
    navigationSections,
    navigateToUnloadedSlot: handleNavigateToUnloadedSlot,
    getSelectionPage,
    onNavigationStart: cancelPendingSparseNavigation,
    selectionMarks,
    selectItemRange,
    toggleItem,
  });

  // The selection a cull key marks; `[` and `]` step from its primary item, the one last added.
  const cullTarget = useMemo((): GalleryCullTarget | null => {
    if (actionSelectionRefs.length === 0) {
      return null;
    }

    const primaryIndex = actionSelectionRefs.findIndex((ref) => toGalleryItemKey(ref) === gallery.selectedItemKey);
    const singleKey = actionSelectionRefs.length === 1 ? toGalleryItemKey(actionSelectionRefs[0]!) : null;

    return {
      item: singleKey === null ? null : (loadedItems.find((item) => toGalleryItemKey(item) === singleKey) ?? null),
      marks: selectionMarks,
      primary: selectionMarks[Math.max(0, primaryIndex)],
      refs: actionSelectionRefs,
    };
  }, [actionSelectionRefs, gallery.selectedItemKey, loadedItems, selectionMarks]);
  const markCullTargets = useCallback(
    (refs: GalleryItemRef[], patch: GalleryMarksPatch) => void itemActions.setItemsMarks(refs, patch, { quiet: true }),
    [itemActions]
  );
  const toggleCullMode = useCallback(
    () => widgets.patchGalleryValues(getGalleryCullModeValues({ enabled: !cullMode.enabled })),
    [cullMode.enabled, widgets]
  );

  const settlePendingSparseNavigation = useEffectEvent(() => {
    const pending = pendingSparseNavigationRef.current;

    if (!pending) {
      return;
    }

    if (pending.filterIdentity !== sparseFilterIdentity || pending.originItemKey !== navigationOriginKey) {
      pendingSparseNavigationRef.current = null;
      return;
    }

    if (pending.shouldRestoreFocus && pending.originFocusKey !== getFocusedTileKey()) {
      pendingSparseNavigationRef.current = null;
      return;
    }

    const item = sparseListing?.itemSlots.get(pending.index);
    const targetPageState = isSparsePaginated
      ? sparseListing?.pageStates.get(sparsePageOffset)
      : sparseListing?.pageStates.get(Math.floor(pending.index / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE);

    // A settled page without the item leaves a gap; a later refresh must not land the step.
    if (!item && targetPageState && !targetPageState.isLoading) {
      pendingSparseNavigationRef.current = null;
      return;
    }

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
  }, [navigationOriginKey, sparseFilterIdentity, sparseListing?.itemSlots, sparseListing?.pageStates]);

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

  const revealRequest = useSyncExternalStore(subscribeGalleryRevealRequests, getGalleryRevealRequest);
  // The index is a slot only in the listing the selection navigates, which may have its own query.
  const selectedImageQuery = getGallerySelectedImageQuery(galleryValues);
  const isRevealIndexInScope =
    isGallerySelectionInScope(selectedImageQuery, gallery) &&
    (selectedImageQuery.semanticKey ?? '') === gallerySemanticReferenceKey(gallery.semanticImageQuery);
  const pendingRevealRef = useRef<GalleryRevealRequest | null>(null);
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

    // The persisted set catches off-page selections whose visible key is null.
    if (
      (gallery.selectedItemKey !== null && gallery.selectedItemKey !== pending.itemKey) ||
      (gallery.selectedItemKeys.length > 0 && !gallery.selectedItemKeys.includes(pending.itemKey))
    ) {
      pendingRevealRef.current = null;

      return;
    }

    // Consumed only once it scrolled, so a row not built yet is honored on the next row-model change.
    if (scrollToItemKey(pending.itemKey)) {
      pendingRevealRef.current = null;

      return;
    }

    // Follow once per reveal so a missing item cannot keep pulling the user back.
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

      if (revealRequest.absoluteIndex !== undefined && isRevealIndexInScope && sparseListing && !isSparsePaginated) {
        const indexedItem = sparseListing.itemSlots.get(revealRequest.absoluteIndex);

        if (!indexedItem || toGalleryItemKey(indexedItem) !== revealRequest.itemKey) {
          requestSparseAbsoluteIndex(revealRequest.absoluteIndex, { verified: true });
        }
      }
    }

    settlePendingReveal();
  }, [
    isRevealIndexInScope,
    isSparsePaginated,
    navigationSections,
    requestSparseAbsoluteIndex,
    revealRequest,
    sparseListing,
  ]);

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

  // Measure before paint, or unchanged visible indices keep stale offsets despite new estimates.
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

          // A missing anchor counts only once its page settled successfully; otherwise it may just be evicted.
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
  // Never the Tab stop kept mounted far below.
  const lastVisibleRowIndex = Math.min((virtualizer.range?.endIndex ?? 0) + GRID_OVERSCAN_ROWS, rowCount - 1);

  useEffect(() => {
    if (!usesSparseListing && paginationMode === 'infinite' && rowCount > 0 && lastVisibleRowIndex >= rowCount - 2) {
      actions.loadMore();
    }
  }, [actions, lastVisibleRowIndex, paginationMode, rowCount, usesSparseListing]);

  const handleDragEnter = useCallback((event: DragEvent) => {
    if (!isFileDragOnGrid(event)) {
      return;
    }

    event.preventDefault();
    dragDepthRef.current += 1;
    setIsDropActive(true);
  }, []);

  const handleDragLeave = useCallback((event: DragEvent) => {
    if (!isFileDragOnGrid(event)) {
      return;
    }

    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

    if (dragDepthRef.current === 0) {
      setIsDropActive(false);
    }
  }, []);

  const handleDragOver = useCallback((event: DragEvent) => {
    if (isFileDragOnGrid(event)) {
      event.preventDefault();
    }
  }, []);

  const handleDrop = useCallback(
    (event: DragEvent) => {
      if (!isEventFromOwnDom(event)) {
        return;
      }
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

  // Show all removes its own header, so focus moves first to the toolbar filter control.
  const handleShowAllStarred = useCallback(() => {
    viewportRef.current
      ?.closest('[role="tabpanel"]')
      ?.parentElement?.querySelector<HTMLElement>('[data-gallery-marks-filter-toggle]')
      ?.focus();
    actions.setMarksFilter({ ...gallery.marksFilter, flags: ['pick'] });
  }, [actions, gallery.marksFilter]);
  // Clear filters removes itself, so focus moves to the toolbar filter control first, as Show all does.
  const handleClearMarksFilter = useCallback(() => {
    viewportRef.current
      ?.closest('[role="tabpanel"]')
      ?.parentElement?.querySelector<HTMLElement>('[data-gallery-marks-filter-toggle]')
      ?.focus();
    actions.setMarksFilter(EMPTY_GALLERY_MARKS_FILTER);
  }, [actions]);

  const handleShowAllMedia = useCallback(() => {
    mediaFilterTriggerRef?.current?.focus();
    actions.updateSettings({ mediaFilter: 'all' });
  }, [actions, mediaFilterTriggerRef]);

  const handleToggleBookmark = useCallback(
    (item: GalleryItem) =>
      void itemActions.setItemsMarks(
        [{ kind: item.kind, name: item.name }],
        toggleGalleryBookmark([getGalleryItemMarks(item)])
      ),
    [itemActions]
  );
  const handleOpenItem = useCallback((item: GalleryItem) => itemActions.openItemInPreview(item), [itemActions]);
  const handleOpenItemInNewTab = useCallback((item: GalleryItem) => itemActions.openItemInNewTab(item), [itemActions]);

  const handleReturnToBoardTop = useCallback(() => galleryCommands.setPage(0), [galleryCommands]);

  // A successful retry unmounts the focused Retry; visible content takes focus without scrolling.
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
              : // While waiting for the second image, the base already carries its side.
                (isComparisonActive || isCompareArmed) && itemKey === gallery.compareImageKey
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
          onOpenInNewTab={middleClickOpensInNewTab ? handleOpenItemInNewTab : null}
          onToggleBookmark={handleToggleBookmark}
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
      handleOpenItemInNewTab,
      handleThumbnailClick,
      handleThumbnailContextMenu,
      handleToggleBookmark,
      isCompareArmed,
      isComparisonActive,
      isFollowingLive,
      middleClickOpensInNewTab,
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
      {isComparisonActive || isCompareArmed ? <GalleryExitCompareHotkey key={`${projectId}:${i18n.language}`} /> : null}
      <GalleryCullCommands
        key={`${projectId}:${i18n.language}:${cullMode.enabled}`}
        autoAdvance={cullMode.autoAdvance}
        enabled={cullMode.enabled}
        marksFilter={gallery.semanticImageQuery === null ? gallery.marksFilter : null}
        runtime={runtime}
        stepper={cullStepper}
        target={cullTarget}
        onMark={markCullTargets}
        onToggle={toggleCullMode}
      />
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
          {/* Focusable from script only, for when the grid has no tiles. */}
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
            {/* Without `flexShrink={0}` the flex viewport squeezes the content and rows overflow their sizer. */}
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
                    <GalleryLoadNotice
                      h={`${GALLERY_STARRED_HEADER_HEIGHT_PX}px`}
                      message={t(
                        starredStrip.state.status === 'error'
                          ? 'widgets.gallery.bookmarkedLoadFailed'
                          : 'widgets.gallery.bookmarkedRefreshFailed'
                      )}
                      px="1"
                      read={starredStrip.state}
                      retryLabel={t(
                        starredStrip.state.status === 'error'
                          ? 'widgets.gallery.retryLoadingBookmarkedItems'
                          : 'widgets.gallery.retryRefreshingBookmarkedItems'
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
                  <LoadError
                    detail={listing.error ? getApiErrorMessage(listing.error, '') : null}
                    flex="1"
                    isRetrying={listing.isRetrying}
                    layout="state"
                    message={t('widgets.gallery.listingLoadFailed')}
                    py="6"
                    retryLabel={t('widgets.gallery.retryLoadingItems')}
                    onFocusLost={focusGridContent}
                    onRetry={listing.retry}
                  />
                </Flex>
              ) : isEmpty && isStarredFailedEmpty && listing.status !== 'loading' ? (
                <Box flex="1" minH="8rem" />
              ) : isEmpty && mediaFilter !== 'all' && listing.status !== 'loading' ? (
                <Stack
                  align="center"
                  color="fg.muted"
                  data-gallery-media-filter-empty
                  flex="1"
                  gap="2"
                  justify="center"
                  minH="8rem"
                >
                  <Text>
                    {t(
                      mediaFilter === 'video' ? 'widgets.gallery.noMatchingVideos' : 'widgets.gallery.noMatchingImages'
                    )}
                  </Text>
                  <Button size="sm" variant="outline" onClick={handleShowAllMedia}>
                    {t('widgets.gallery.showAllMedia')}
                  </Button>
                </Stack>
              ) : isEmpty && hasMarksFilter && listing.status !== 'loading' ? (
                <Stack
                  align="center"
                  color="fg.muted"
                  data-gallery-marks-filter-empty
                  flex="1"
                  gap="2"
                  justify="center"
                  minH="8rem"
                >
                  <Text>{t('widgets.gallery.marksFilter.noMatches')}</Text>
                  <Button size="sm" variant="outline" onClick={handleClearMarksFilter}>
                    {t('widgets.gallery.marksFilter.clearFilters')}
                  </Button>
                </Stack>
              ) : isEmpty ? (
                listing.status === 'loading' || hasActiveSearch || isVirtualBoard ? (
                  <Flex align="center" color="fg.muted" flex="1" justify="center" minH="8rem">
                    <Text>
                      {listing.status === 'loading'
                        ? t('widgets.gallery.loadingBackendGallery')
                        : t('widgets.gallery.noImagesMatch')}
                    </Text>
                  </Flex>
                ) : (
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
                  {/* Fixed pitch: the virtualizer's total trails a row-count change by a commit. */}
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
