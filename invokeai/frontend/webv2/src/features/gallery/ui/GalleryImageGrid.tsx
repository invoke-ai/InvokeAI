import { Box, chakra, Flex, HStack, Icon, ScrollArea, Spinner, Stack, Text } from '@chakra-ui/react';
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
import { defaultRangeExtractor, useVirtualizer, type Range } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import {
  buildGalleryGridRows,
  chunkGalleryCellsIntoRows,
  GALLERY_GRID_GAP_PX,
  GALLERY_PINNED_FOOTER_PX,
  GALLERY_STARRED_HEADER_HEIGHT_PX,
  getGalleryCellSizePx,
  getGalleryColumnCount,
  getGalleryGridRowIndexForItemKey,
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

// Module-scoped so a grid remount cannot replay an already-followed reveal.
let lastPageFollowedRevealToken = 0;

const dragEventContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

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
const extractRangeKeeping = (range: Range, keptRow: number): number[] => {
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
  viewport.querySelector<HTMLElement>(
    entry.kind === 'session'
      ? `[data-gallery-session-id="${CSS.escape(entry.id)}"]`
      : `[data-gallery-item-key="${CSS.escape(toGalleryItemKey(entry.item))}"]`
  );

/** Prefers a visible tile; an empty grid offers whatever else it shows (its upload target, the strip header). */
const focusVisibleGridContent = (viewport: HTMLElement | null, edge: 'first' | 'last') => {
  if (!focusVisibleOperable(viewport, { edge, selector: TILE_BUTTON_SELECTOR })) {
    focusVisibleOperable(viewport, { edge });
  }
};

/** Measure viewport width for columns so both layouts share the same grid. */
export const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const { actions, filter, gallery, isWindowTruncated, itemActions, listing, region, starredStrip } =
    useGalleryWidget();
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
  } = useGalleryGridSelection();

  const columnCount = getGalleryColumnCount({ imageDensityPercent, widthPx: viewportWidth });
  const isFollowingLive = followedProgressSessionId !== null;
  const isComparisonActive = gallery.isComparisonActive && !isFollowingLive;
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const selectedBoardName = selectedBoard
    ? getGalleryBoardLabel(selectedBoard, t)
    : t('widgets.gallery.selectedBoardFallback');
  // The listing is unstarred-only, so a board whose items are all starred
  // still has the strip to show.
  const isEmpty = gallery.items.length === 0 && starredStrip.items.length === 0;
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

    return [
      stripEntries,
      isProgressOpen
        ? progressSessions.map((session) => ({
            id: session.id,
            kind: 'session',
            navigable: session.state === 'running',
          }))
        : [],
      gallery.items.map((item) => ({ item, kind: 'item' })),
    ];
  }, [gallery.items, hiddenStripSelection, isProgressOpen, progressSessions, shownStripItems]);

  // Thumbnails in visual order, strip first. They share one thumbnail Tab stop: the tile focus was last on, else the
  // selection, else the tile that took the focused one's place in the same scope (after a deletion, the neighbour
  // selected next), else the first. Progress tiles, disclosures, the upload zone and retries keep their own stops.
  const tileKeys = useMemo(
    () => [...shownStripItems, ...gallery.items].map(toGalleryItemKey),
    [gallery.items, shownStripItems]
  );
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
  const tabStopRow =
    tabStopIndex < shownStripItems.length ? -1 : Math.floor((tabStopIndex - shownStripItems.length) / columnCount);
  // Scrolling never unmounts the Tab stop, which is also the focused tile while the grid holds focus: focus would
  // fall to the document body, and Tab would skip the grid.
  const rangeExtractor = useCallback((range: Range) => extractRangeKeeping(range, tabStopRow), [tabStopRow]);

  const rowCount = rows.length;
  const cellSizePx = getGalleryCellSizePx({ columnCount, widthPx: viewportWidth });
  const rowHeightPx = cellSizePx + GALLERY_GRID_GAP_PX;
  const estimateRowSize = useCallback(() => rowHeightPx, [rowHeightPx]);
  const getRowKey = useCallback((index: number) => rows[index]?.key ?? index, [rows]);
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
  const virtualizer = useVirtualizer({
    count: rowCount,
    scrollMargin: pinnedHeight,
    estimateSize: estimateRowSize,
    getItemKey: getRowKey,
    getScrollElement,
    overscan: GRID_OVERSCAN_ROWS,
    rangeExtractor,
  });

  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  // The hotkey callback reads current state when invoked; stabilizing its identity adds no value and interferes
  // with compiler memoization.
  /** Returns whether the item had somewhere to scroll to — a collapsed strip has none. */
  const scrollToItemKey = (itemKey: GalleryItemKey): boolean => {
    const rowIndex = getGalleryGridRowIndexForItemKey(gallery.items, itemKey, columnCount);

    if (rowIndex >= 0) {
      virtualizer.scrollToIndex(rowIndex);
      return true;
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
    selectItemRange,
    toggleItem,
  });

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
    if (revealRequest && revealRequest.token !== consumedRevealTokenRef.current) {
      consumedRevealTokenRef.current = revealRequest.token;
      pendingRevealRef.current = revealRequest;
    }

    settlePendingReveal();
  }, [revealRequest, navigationSections]);

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
  useLayoutEffect(() => {
    measureVirtualizer();
  }, [rowHeightPx, rows, pinnedHeight]);

  const virtualRows = virtualizer.virtualItems;
  // From the rows in view and their overscan, never the Tab stop kept mounted far below them.
  const lastVisibleRowIndex = Math.min((virtualizer.range?.endIndex ?? 0) + GRID_OVERSCAN_ROWS, rowCount - 1);

  useEffect(() => {
    if (paginationMode === 'infinite' && rowCount > 0 && lastVisibleRowIndex >= rowCount - 2) {
      actions.loadMore();
    }
  }, [actions, lastVisibleRowIndex, paginationMode, rowCount]);

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
    (item: GalleryItem) => {
      const itemKey = toGalleryItemKey(item);

      return (
        <GalleryThumbnailCell
          key={itemKey}
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

  return (
    <Stack flex="1" gap="0" h="full" minH="0" minW="0" w="full">
      {listing.status === 'stale-error' ? (
        <GalleryLoadNotice
          bg="bg.panel"
          flexShrink={0}
          message={t('widgets.gallery.listingRefreshFailed')}
          px="2"
          py="1"
          read={listing}
          retryLabel={t('widgets.gallery.retryLoadingItems')}
          onFocusLost={focusGridContent}
        />
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
        {gallery.anchoredWindowPage > 0 ? (
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
                // A failed scope never reads as an empty board, a search with no matches, or another scope's items.
                <Flex flex="1" minH="8rem">
                  <GalleryLoadErrorState
                    read={listing}
                    retryLabel={t('widgets.gallery.retryLoadingItems')}
                    title={t('widgets.gallery.listingLoadFailed')}
                    onFocusLost={focusGridContent}
                  />
                </Flex>
              ) : isEmpty && isStarredFailedEmpty && listing.status !== 'loading' ? (
                // The unstarred listing is empty, but the board's starred items are unknown: neither "empty board"
                // nor "no matches" would be true, so only the strip's failure speaks.
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
                            {row.cells.map(renderCell)}
                          </Box>
                        );
                      })}
                    </Box>
                  </Box>
                  {listing.status === 'more-error' ? (
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
                  ) : paginationMode === 'infinite' &&
                    (listing.isFetchingMore || listing.status === 'loading') &&
                    gallery.items.length > 0 ? (
                    <Flex align="center" justify="center" py="2">
                      <Spinner color="fg.subtle" />
                    </Flex>
                  ) : null}
                  {paginationMode === 'infinite' && !listing.isFetchingMore && isWindowTruncated && (
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
