import { Box, Flex, HStack, Icon, ScrollArea, Spinner, Text } from '@chakra-ui/react';
import { getGalleryBoardLabel } from '@features/gallery/core/boardLabels';
import { toGalleryItemKey, type GalleryItem } from '@features/gallery/core/items';
import { Button, DropZone } from '@platform/ui';
import { ChevronRightIcon, StarIcon, UploadIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GalleryQueuePlaceholder } from './galleryStateView';

import { GALLERY_GRID_GAP_PX, getGalleryCellSizePx, getGalleryColumnCount } from './galleryGridLayout';
import { GalleryQueuePlaceholderCell } from './GalleryQueuePlaceholderCell';
import { getGalleryPlaceholderInsertionIndex } from './galleryStateView';
import { GalleryThumbnailCell } from './GalleryThumbnail';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';
import { useGalleryGridHotkeys } from './useGalleryGridHotkeys';
import { useGalleryGridSelection } from './useGalleryGridSelection';

/**
 * Seeds the measured width per region so remounting the gallery in a placement
 * it has already been shown in does not paint one frame of fallback-sized
 * tiles before the ResizeObserver reports.
 */
const viewportWidthCache = new Map<string, number>();
const STARRED_HEADER_HEIGHT_PX = 28;

const dragEventContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

type GridCell =
  | { kind: 'item'; item: GalleryItem; itemIndex: number }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

type GridSection = 'regular' | 'starred';

type GridRow =
  | { kind: 'cells'; cells: GridCell[]; section: GridSection }
  | { kind: 'starred-header'; itemCount: number };

const chunkCellsIntoRows = (cells: GridCell[], columnCount: number, section: GridSection): GridRow[] => {
  const rows: GridRow[] = [];

  for (let index = 0; index < cells.length; index += columnCount) {
    rows.push({ cells: cells.slice(index, index + columnCount), kind: 'cells', section });
  }

  return rows;
};

/**
 * The virtualized thumbnail grid. Column count comes from the measured
 * viewport width, so the grid is layout-blind: both shells render it the same
 * way and it simply fills whatever box it is handed.
 */
export const GalleryImageGrid = () => {
  const { t } = useTranslation();
  const { actions, gallery, isWindowTruncated, itemActions, region } = useGalleryWidget();
  const { account, antialiasProgressImages, ImageContextMenu } = useGalleryUi();
  const [isDropActive, setIsDropActive] = useState(false);
  const [isStarredOpen, setIsStarredOpen] = useState(true);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(region) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, thumbnailFit } = gallery.settings;

  const {
    actionSelectionRefs,
    activeContextMenuTarget,
    getDragItems,
    handleCloseContextMenu,
    handleThumbnailClick,
    handleThumbnailContextMenu,
    selectedItemKeys,
    syncRangeInteractionContext,
  } = useGalleryGridSelection();

  const columnCount = getGalleryColumnCount({ imageDensityPercent, widthPx: viewportWidth });
  const isFollowingLive = gallery.currentItem?.kind === 'placeholder';
  const isComparisonActive =
    gallery.selectedItemKey?.startsWith('image:') === true &&
    gallery.compareImageKey !== null &&
    gallery.selectedItemKey !== null &&
    gallery.compareImageKey !== gallery.selectedItemKey;
  const selectedBoard = gallery.boards.find((board) => board.id === gallery.selectedBoardId);
  const selectedBoardName = selectedBoard
    ? getGalleryBoardLabel(selectedBoard, t)
    : t('widgets.gallery.selectedBoardFallback');
  const isEmpty = gallery.items.length === 0 && gallery.pendingPlaceholders.length === 0;

  const rows = useMemo<GridRow[]>(() => {
    const regularItemCells: GridCell[] = [];
    const regularItems: GalleryItem[] = [];
    const starredItemCells: GridCell[] = [];

    gallery.items.forEach((item, itemIndex) => {
      const cell: GridCell = { item, itemIndex, kind: 'item' };

      if (item.starred) {
        starredItemCells.push(cell);
      } else {
        regularItemCells.push(cell);
        regularItems.push(item);
      }
    });

    const placeholderCells: GridCell[] = gallery.pendingPlaceholders.map((placeholder) => ({
      kind: 'placeholder',
      placeholder,
    }));
    const placeholderInsertionIndex = getGalleryPlaceholderInsertionIndex(regularItems, imageOrderDir, false);
    const regularCells = [
      ...regularItemCells.slice(0, placeholderInsertionIndex),
      ...placeholderCells,
      ...regularItemCells.slice(placeholderInsertionIndex),
    ];
    const nextRows: GridRow[] = [];

    if (starredItemCells.length > 0) {
      nextRows.push({ itemCount: starredItemCells.length, kind: 'starred-header' });

      if (isStarredOpen) {
        nextRows.push(...chunkCellsIntoRows(starredItemCells, columnCount, 'starred'));
      }
    }

    nextRows.push(...chunkCellsIntoRows(regularCells, columnCount, 'regular'));

    return nextRows;
  }, [columnCount, gallery.items, gallery.pendingPlaceholders, imageOrderDir, isStarredOpen]);

  const rowCount = rows.length;
  const cellSizePx = getGalleryCellSizePx({ columnCount, widthPx: viewportWidth });
  const rowHeightPx = cellSizePx + GALLERY_GRID_GAP_PX;
  const estimateRowSize = useCallback(
    (index: number) => (rows[index]?.kind === 'starred-header' ? STARRED_HEADER_HEIGHT_PX : rowHeightPx),
    [rowHeightPx, rows]
  );
  const getScrollElement = useCallback(() => viewportRef.current, []);

  const virtualizer = useVirtualizer({
    count: rowCount,
    estimateSize: estimateRowSize,
    getScrollElement,
    overscan: 4,
  });

  const measureVirtualizer = useEffectEvent(() => {
    virtualizer.measure();
  });

  // Deliberately unmemoized: the hotkey hook only ever calls this from inside
  // an effect event, which always sees the latest render's closure, so a stable
  // identity would buy nothing and pinning the mutable virtualizer into a
  // dependency array defeats the compiler's own memoization.
  const scrollToItemIndex = (itemIndex: number) => {
    const rowIndex = rows.findIndex(
      (row) => row.kind === 'cells' && row.cells.some((cell) => cell.kind === 'item' && cell.itemIndex === itemIndex)
    );

    if (rowIndex >= 0) {
      virtualizer.scrollToIndex(rowIndex);
    }
  };

  useGalleryGridHotkeys({ actionSelectionRefs, columnCount, scrollToItemIndex });

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

  useEffect(() => {
    measureVirtualizer();
  }, [rowHeightPx]);

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

  const handleShowProgressImages = useCallback(() => {
    account.enableLiveFollow();
  }, [account]);

  const handleToggleStarredSection = useCallback(() => {
    setIsStarredOpen((isOpen) => !isOpen);
  }, []);

  const handleToggleStarred = useCallback(
    (item: GalleryItem) => void itemActions.setItemsStarred([{ kind: item.kind, name: item.name }], !item.starred),
    [itemActions]
  );

  return (
    <Box
      ref={syncRangeInteractionContext}
      flex="1"
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
      {isEmpty ? (
        <Flex align="center" color="fg.muted" h="full" justify="center" minH="8rem">
          <Text fontSize="xs">
            {gallery.isLoading ? t('widgets.gallery.loadingBackendGallery') : t('widgets.gallery.noImagesMatch')}
          </Text>
        </Flex>
      ) : (
        <ScrollArea.Root h="full" minH="0" size="xs" variant="hover" w="full">
          <ScrollArea.Viewport ref={viewportRef} h="full" outline="none" w="full">
            <ScrollArea.Content aria-label={t('widgets.gallery.itemsAriaLabel')} role="list">
              <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
                {virtualRows.map((virtualRow) => {
                  const row = rows[virtualRow.index];

                  if (!row) {
                    return null;
                  }

                  if (row.kind === 'starred-header') {
                    return (
                      <Flex
                        key={virtualRow.key}
                        align="center"
                        h={`${STARRED_HEADER_HEIGHT_PX}px`}
                        left="0"
                        position="absolute"
                        top="0"
                        transform={`translateY(${virtualRow.start}px)`}
                        w="full"
                      >
                        <Button
                          aria-expanded={isStarredOpen}
                          aria-label={t(
                            isStarredOpen
                              ? 'widgets.gallery.collapseStarredItems'
                              : 'widgets.gallery.expandStarredItems'
                          )}
                          color="fg.muted"
                          h="6"
                          justifyContent="flex-start"
                          px="1"
                          size="xs"
                          variant="ghost"
                          w="full"
                          onClick={handleToggleStarredSection}
                        >
                          <Icon
                            as={ChevronRightIcon}
                            boxSize="3"
                            transform={isStarredOpen ? 'rotate(90deg)' : undefined}
                            transition="transform var(--wb-motion-duration-medium) ease"
                          />
                          <Icon as={StarIcon} boxSize="3" fill="currentColor" />
                          <HStack gap="1">
                            <Text
                              as="span"
                              fontSize="2xs"
                              fontWeight="600"
                              letterSpacing="wide"
                              textTransform="uppercase"
                            >
                              {t('widgets.gallery.starredItems')}
                            </Text>
                            <Text as="span" color="fg.subtle" fontSize="2xs" fontVariantNumeric="tabular-nums">
                              {row.itemCount}
                            </Text>
                          </HStack>
                        </Button>
                      </Flex>
                    );
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
                      transform={`translateY(${virtualRow.start}px)`}
                      w="full"
                    >
                      {row.cells.map((cell) =>
                        cell.kind === 'placeholder' ? (
                          <GalleryQueuePlaceholderCell
                            key={cell.placeholder.id}
                            antialiasProgressImages={antialiasProgressImages}
                            fit={thumbnailFit}
                            isSelected={
                              gallery.currentItem?.kind === 'placeholder' &&
                              gallery.currentItem.placeholder.id === cell.placeholder.id
                            }
                            placeholder={cell.placeholder}
                            onClick={handleShowProgressImages}
                          />
                        ) : (
                          <GalleryThumbnailCell
                            key={toGalleryItemKey(cell.item)}
                            alwaysShowDimensions={showImageDimensions}
                            dragScope={region}
                            compareRole={
                              isComparisonActive && toGalleryItemKey(cell.item) === gallery.selectedItemKey
                                ? t('widgets.preview.viewing')
                                : isComparisonActive && toGalleryItemKey(cell.item) === gallery.compareImageKey
                                  ? t('widgets.preview.compare')
                                  : null
                            }
                            fit={thumbnailFit}
                            getDragItems={getDragItems}
                            isPrimary={!isFollowingLive && toGalleryItemKey(cell.item) === gallery.selectedItemKey}
                            isSelected={!isFollowingLive && selectedItemKeys.has(toGalleryItemKey(cell.item))}
                            item={cell.item}
                            onClick={handleThumbnailClick}
                            onContextMenu={handleThumbnailContextMenu}
                            onToggleStarred={handleToggleStarred}
                          />
                        )
                      )}
                    </Box>
                  );
                })}
              </Box>
              {paginationMode === 'infinite' && gallery.isLoading && gallery.items.length > 0 && (
                <Flex align="center" justify="center" py="2">
                  <Spinner color="fg.subtle" size="xs" />
                </Flex>
              )}
              {paginationMode === 'infinite' && !gallery.isLoading && isWindowTruncated && (
                <Flex align="center" justify="center" py="3">
                  <Text color="fg.subtle" fontSize="xs" textAlign="center">
                    {t('widgets.gallery.windowLimit', { count: gallery.items.length })}
                  </Text>
                </Flex>
              )}
            </ScrollArea.Content>
          </ScrollArea.Viewport>
          <ScrollArea.Scrollbar>
            <ScrollArea.Thumb />
          </ScrollArea.Scrollbar>
        </ScrollArea.Root>
      )}
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
  );
};
