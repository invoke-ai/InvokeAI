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

// Module-scoped so a grid remount cannot replay an already-followed reveal.
let lastPageFollowedRevealToken = 0;

const dragEventContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

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
            fontSize="2xs"
            fontWeight="600"
            letterSpacing="wide"
            lineHeight="1"
            textTransform="uppercase"
            truncate
          >
            {t('widgets.gallery.starredItems')}
          </Text>
          <Text as="span" color="currentColor" fontSize="2xs" fontVariantNumeric="tabular-nums" lineHeight="1">
            {total}
          </Text>
        </HStack>
      </chakra.button>
      {total > shownCount ? (
        <Button
          aria-label={t('widgets.gallery.showAllStarredItems')}
          color="fg.muted"
          flexShrink={0}
          size="2xs"
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
  const { actions, gallery, isWindowTruncated, itemActions, region, starredStrip } = useGalleryWidget();
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
  const isFollowingLive = followedProgressSessionId !== null;
  const isComparisonActive = gallery.isComparisonActive && !isFollowingLive;
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const selectedBoardName = selectedBoard
    ? getGalleryBoardLabel(selectedBoard, t)
    : t('widgets.gallery.selectedBoardFallback');
  // The listing is unstarred-only, so a board whose items are all starred
  // still has the strip to show.
  const isEmpty = gallery.items.length === 0 && starredStrip.items.length === 0;
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
  }, [
    gallery.items,
    gallery.selectedItemKey,
    isProgressOpen,
    isStarredOpen,
    progressSessions,
    starredCells,
    starredStrip.items,
  ]);
  const cursorKey =
    followedProgressSessionId !== null
      ? getGallerySessionNavigationKey(followedProgressSessionId)
      : gallery.selectedItemKey;

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
    overscan: 4,
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

  useGalleryGridHotkeys({
    actionSelectionRefs,
    columnCount,
    cursorKey,
    loadedItems,
    navigationSections,
    scrollToEntry,
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
  const lastVisibleRowIndex = virtualRows[virtualRows.length - 1]?.index ?? 0;

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

  // Releasing the anchor puts the window back over the top of the listing.
  const handleReturnToBoardTop = useCallback(() => galleryCommands.setPage(0), [galleryCommands]);

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
          item={item}
          onClick={handleThumbnailClick}
          onContextMenu={handleThumbnailContextMenu}
          onToggleStarred={handleToggleStarred}
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

  return (
    <Stack flex="1" gap="0" h="full" minH="0" minW="0" w="full">
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
            <Text color="fg.muted" fontSize="2xs" truncate>
              {t('widgets.gallery.windowAnchored', { index: anchoredWindowFirstItem })}
            </Text>
            <Button flexShrink={0} size="2xs" variant="ghost" onClick={handleReturnToBoardTop}>
              {t('widgets.gallery.backToBoardTop')}
            </Button>
          </Flex>
        ) : null}
        <ScrollArea.Root h="full" minH="0" size="xs" variant="hover" w="full">
          <ScrollArea.Viewport ref={viewportRef} h="full" outline="none" w="full">
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
                gallery.isLoading || hasActiveSearch || isVirtualBoard || gallery.starredOnly ? (
                  <Flex align="center" color="fg.muted" flex="1" justify="center" minH="8rem">
                    <Text fontSize="xs">
                      {gallery.isLoading
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
                      fontSize="xs"
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
                  {paginationMode === 'infinite' && gallery.isLoading && gallery.items.length > 0 && (
                    <Flex align="center" justify="center" py="2">
                      <Spinner color="fg.subtle" size="xs" />
                    </Flex>
                  )}
                  {paginationMode === 'infinite' && !gallery.isLoading && isWindowTruncated && (
                    <Flex align="center" justify="center" py="3">
                      <Text color="fg.subtle" fontSize="xs" textAlign="center">
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
            <Text fontSize="xs" fontWeight="600">
              {t('widgets.gallery.dropMediaToUploadToBoard', { name: selectedBoardName })}
            </Text>
          </DropZone>
        )}
        <ImageContextMenu boards={gallery.boards} target={activeContextMenuTarget} onClose={handleCloseContextMenu} />
      </Box>
    </Stack>
  );
};
