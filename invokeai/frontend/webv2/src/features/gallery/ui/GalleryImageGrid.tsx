import type { GalleryItem, GalleryItemRef } from '@features/gallery/core/items';
import type { GalleryThumbnailFit } from '@features/gallery/core/settings';

import { Badge, Box, Flex, ProgressCircle, ScrollArea, Skeleton, Spinner, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import {
  formatGalleryVideoDuration,
  isGalleryImageItem,
  parseGalleryItemKey,
  toGalleryItemKey,
  toGalleryItemRef,
} from '@features/gallery/core/items';
import { isDateBoardId, type GalleryItemNames } from '@features/gallery/data/backend';
import { galleryItemNamesOptions, type GalleryItemsFilter } from '@features/gallery/data/queries';
import { useQueueItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { parseDateTokens } from '@platform/search/dateTokens';
import { captureAccountScope, isAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { DropZone, IconButton } from '@platform/ui';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { useQueryClient } from '@tanstack/react-query';
import { PlayIcon, StarIcon, UploadIcon } from 'lucide-react';
import {
  useCallback,
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent,
  type MouseEvent,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GalleryQueuePlaceholder } from './galleryStateView';
import type { GalleryItemContextMenuTarget } from './GalleryUiContext';

import { getGalleryItemDragData, getGalleryItemDragId } from './galleryDnd';
import { getGalleryPlaceholderInsertionIndex } from './galleryStateView';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

const GRID_GAP_PX = 4;
const THUMBNAIL_HOVER_CSS = {
  '&:hover .gallery-thumb-overlay, &:focus-within .gallery-thumb-overlay': { opacity: 1 },
  '&:focus-within': { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' },
} as const;
const THUMBNAIL_BUTTON_STYLE = {
  background: 'transparent',
  border: 0,
  cursor: 'pointer',
  display: 'block',
  height: '100%',
  inset: 0,
  minWidth: 0,
  outline: 'none',
  padding: 0,
  position: 'absolute',
  width: '100%',
} as const;

const viewportWidthCache = new Map<'stacked' | 'wide', number>();

const getGalleryColumnCount = (imageDensityPercent: number, layout: 'stacked' | 'wide'): number => {
  const min = 2;
  const max = layout === 'wide' ? 10 : 6;
  const percent = Math.min(100, Math.max(0, imageDensityPercent));

  return Math.round(min + ((max - min) * percent) / 100);
};

const dragEventContainsFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer.types).includes('Files');

const getGalleryItemRange = (
  orderedRefs: readonly GalleryItemRef[],
  anchorItemKey: string,
  targetItemKey: string
): GalleryItemRef[] | null => {
  const anchorIndex = orderedRefs.findIndex((ref) => toGalleryItemKey(ref) === anchorItemKey);
  const targetIndex = orderedRefs.findIndex((ref) => toGalleryItemKey(ref) === targetItemKey);

  if (anchorIndex === -1 || targetIndex === -1) {
    return null;
  }

  const start = Math.min(anchorIndex, targetIndex);
  const end = Math.max(anchorIndex, targetIndex);

  return orderedRefs.slice(start, end + 1);
};

type GridCell =
  | { kind: 'item'; item: GalleryItem; itemIndex: number }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

export const GalleryImageGrid = ({ layout }: { layout: 'stacked' | 'wide' }) => {
  const { t } = useTranslation();
  const { actions, gallery, isWindowTruncated, itemActions, runtime } = useGalleryWidget();
  const { account, antialiasProgressImages, ImageContextMenu, widgets } = useGalleryUi();
  const queryClient = useQueryClient();
  const [contextMenuTarget, setContextMenuTarget] = useState<GalleryItemContextMenuTarget | null>(null);
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(layout) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;
  const columnCount = getGalleryColumnCount(imageDensityPercent, layout);
  const selectedItemKeys = useMemo(() => new Set(gallery.selectedItemKeys), [gallery.selectedItemKeys]);
  const selectedItemRefs = useMemo(() => gallery.selectedItemKeys.map(parseGalleryItemKey), [gallery.selectedItemKeys]);
  const isFollowingLive = gallery.currentItem?.kind === 'placeholder';
  const isComparisonActive =
    gallery.selectedItemKey?.startsWith('image:') === true &&
    gallery.compareImageKey !== null &&
    gallery.selectedItemKey !== null &&
    gallery.compareImageKey !== gallery.selectedItemKey;
  const selectedBoardName =
    gallery.boards.find((board) => board.id === gallery.selectedBoardId)?.name ??
    t('widgets.gallery.selectedBoardFallback');
  const isEmpty = gallery.items.length === 0 && gallery.pendingPlaceholders.length === 0;

  const placeholderInsertionIndex = getGalleryPlaceholderInsertionIndex(gallery.items, imageOrderDir, starredFirst);

  const cells = useMemo<GridCell[]>(() => {
    const itemCells: GridCell[] = gallery.items.map((item, itemIndex) => ({
      item,
      itemIndex,
      kind: 'item',
    }));
    const placeholderCells: GridCell[] = gallery.pendingPlaceholders.map((placeholder) => ({
      kind: 'placeholder',
      placeholder,
    }));

    return [
      ...itemCells.slice(0, placeholderInsertionIndex),
      ...placeholderCells,
      ...itemCells.slice(placeholderInsertionIndex),
    ];
  }, [gallery.items, gallery.pendingPlaceholders, placeholderInsertionIndex]);

  const getCellIndexForItem = (itemIndex: number): number =>
    itemIndex >= placeholderInsertionIndex ? itemIndex + gallery.pendingPlaceholders.length : itemIndex;

  const rowCount = Math.ceil(cells.length / columnCount);

  const cellSizePx =
    viewportWidth > 0 ? Math.max(1, (viewportWidth - GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;

  const rowHeightPx = cellSizePx + GRID_GAP_PX;
  const estimateRowSize = useCallback(() => rowHeightPx, [rowHeightPx]);
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

  useLayoutEffect(() => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return;
    }

    const setMeasuredWidth = (width: number) => {
      if (width <= 0) {
        return;
      }

      viewportWidthCache.set(layout, width);
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
  }, [isEmpty, layout]);

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

  const parsedSearch = useMemo(() => parseDateTokens(gallery.searchTerm), [gallery.searchTerm]);
  const itemNamesFilter = useMemo<GalleryItemsFilter>(
    () => ({
      boardId: gallery.selectedBoardId,
      ...(parsedSearch.range?.from ? { createdFrom: parsedSearch.range.from } : {}),
      ...(parsedSearch.range?.to ? { createdTo: parsedSearch.range.to } : {}),
      galleryView: gallery.galleryView,
      orderDir: imageOrderDir,
      searchTerm: parsedSearch.text,
      starredFirst,
    }),
    [gallery.galleryView, gallery.selectedBoardId, imageOrderDir, parsedSearch.range, parsedSearch.text, starredFirst]
  );
  const itemNamesFilterIdentity = useMemo(() => JSON.stringify(itemNamesFilter), [itemNamesFilter]);
  const rangeInteractionContextRef = useRef({
    filterIdentity: itemNamesFilterIdentity,
    selectedItemKey: gallery.selectedItemKey,
  });
  const syncRangeInteractionContext = useCallback(
    (node: HTMLDivElement | null) => {
      if (node) {
        rangeInteractionContextRef.current = {
          filterIdentity: itemNamesFilterIdentity,
          selectedItemKey: gallery.selectedItemKey,
        };
      }
    },
    [gallery.selectedItemKey, itemNamesFilterIdentity]
  );

  const activeContextMenuTarget = useMemo(() => {
    const primaryTargetItem = contextMenuTarget?.items[0];

    if (!primaryTargetItem) {
      return contextMenuTarget;
    }

    const primaryTargetItemKey = toGalleryItemKey(primaryTargetItem);

    return gallery.items.some((item) => toGalleryItemKey(item) === primaryTargetItemKey) ? contextMenuTarget : null;
  }, [contextMenuTarget, gallery.items]);

  const selectItemRange = useCallback(
    async (item: GalleryItem) => {
      const owner = captureAccountScope();
      const capturedContext = rangeInteractionContextRef.current;
      const anchorItemKey = capturedContext.selectedItemKey;
      const targetItemKey = toGalleryItemKey(item);

      if (!anchorItemKey) {
        actions.selectItem(item);
        return;
      }

      const isInteractionCurrent = () =>
        isAccountScopeCurrent(owner) &&
        rangeInteractionContextRef.current.filterIdentity === capturedContext.filterIdentity &&
        rangeInteractionContextRef.current.selectedItemKey === capturedContext.selectedItemKey;
      const selectFromRefs = (refs: readonly GalleryItemRef[]): boolean => {
        const range = getGalleryItemRange(refs, anchorItemKey, targetItemKey);

        if (!range) {
          return false;
        }

        actions.selectItemRange(range, item);
        return true;
      };
      const materializedRefs = gallery.items.map(toGalleryItemRef);
      const namesOptions = galleryItemNamesOptions(itemNamesFilter);

      try {
        const orderedRefs = isDateBoardId(itemNamesFilter.boardId)
          ? queryClient.getQueryData<GalleryItemNames>(namesOptions.queryKey)?.items
          : (await queryClient.fetchQuery(namesOptions)).items;

        if (!isInteractionCurrent()) {
          return;
        }

        if (orderedRefs && selectFromRefs(orderedRefs)) {
          return;
        }
      } catch {
        if (!isInteractionCurrent()) {
          return;
        }
      }

      if (!selectFromRefs(materializedRefs)) {
        actions.selectItem(item);
      }
    },
    [actions, gallery.items, itemNamesFilter, queryClient]
  );

  const handleThumbnailClick = useCallback(
    (item: GalleryItem, event: MouseEvent) => {
      if (event.shiftKey) {
        void selectItemRange(item);
        return;
      }

      if (event.altKey && isGalleryImageItem(item)) {
        actions.setCompareItem(item);
        return;
      }

      if (event.ctrlKey || event.metaKey) {
        const itemKey = toGalleryItemKey(item);
        const remainingItemKeys = gallery.selectedItemKeys.filter((key) => key !== itemKey);
        const nextPrimaryItem =
          gallery.selectedItemKey === itemKey
            ? (gallery.items.find(
                (candidate) => toGalleryItemKey(candidate) === remainingItemKeys[remainingItemKeys.length - 1]
              ) ?? null)
            : null;

        actions.toggleItemInSelection(item, nextPrimaryItem);
      } else {
        actions.selectItem(item);
      }
    },
    [actions, gallery.items, gallery.selectedItemKey, gallery.selectedItemKeys, selectItemRange]
  );

  const handleThumbnailContextMenu = useCallback(
    (item: GalleryItem, x: number, y: number) => {
      const itemKey = toGalleryItemKey(item);

      if (selectedItemKeys.has(itemKey) && selectedItemKeys.size > 1) {
        const itemRefs = selectedItemRefs;
        const selectionItems = [
          item,
          ...gallery.items.filter(
            (candidate) => toGalleryItemKey(candidate) !== itemKey && selectedItemKeys.has(toGalleryItemKey(candidate))
          ),
        ];

        setContextMenuTarget({ itemRefs, items: selectionItems, x, y });
        return;
      }

      setContextMenuTarget({ itemRefs: [toGalleryItemRef(item)], items: [item], x, y });
    },
    [gallery.items, selectedItemKeys, selectedItemRefs]
  );

  const getDragItems = useCallback(
    (item: GalleryItem): GalleryItemRef[] => {
      const itemKey = toGalleryItemKey(item);

      if (selectedItemKeys.has(itemKey) && selectedItemKeys.size > 1) {
        return selectedItemRefs;
      }

      return [toGalleryItemRef(item)];
    },
    [selectedItemKeys, selectedItemRefs]
  );

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

  const handleToggleStarred = useCallback(
    (item: GalleryItem) => void itemActions.setItemsStarred([toGalleryItemRef(item)], !item.starred),
    [itemActions]
  );

  const handleCloseContextMenu = useCallback(() => setContextMenuTarget(null), []);

  const actionSelectionRefs = useMemo(
    () =>
      selectedItemRefs.length > 0
        ? selectedItemRefs
        : gallery.selectedItemKey
          ? [parseGalleryItemKey(gallery.selectedItemKey)]
          : [],
    [gallery.selectedItemKey, selectedItemRefs]
  );

  const navigate = useEffectEvent((direction: 'down' | 'left' | 'right' | 'up') => {
    const itemCount = gallery.items.length;

    if (itemCount === 0) {
      return;
    }

    const selectedIndex = gallery.items.findIndex((item) => toGalleryItemKey(item) === gallery.selectedItemKey);
    const fallbackIndex = selectedIndex === -1 ? 0 : selectedIndex;
    const delta =
      direction === 'right' ? 1 : direction === 'left' ? -1 : direction === 'down' ? columnCount : -columnCount;
    const nextIndex = Math.min(itemCount - 1, Math.max(0, fallbackIndex + delta));
    const nextItem = gallery.items[nextIndex];

    if (nextItem && (nextIndex !== selectedIndex || selectedIndex === -1)) {
      actions.selectItem(nextItem);
      virtualizer.scrollToIndex(Math.floor(getCellIndexForItem(nextIndex) / columnCount));
    }
  });

  const executeGalleryHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'gallery.selectAllOnPage') {
      const primaryItem = gallery.items[0];

      if (primaryItem) {
        actions.selectItemRange(gallery.items.map(toGalleryItemRef), primaryItem);
      }
      return;
    }

    if (commandId === 'gallery.clearSelection') {
      widgets.patchGalleryValues({
        selectedImage: null,
        selectedImageName: null,
        selectedImageNames: [],
      });
      return;
    }

    if (commandId === 'gallery.deleteSelection' && actionSelectionRefs.length > 0) {
      void itemActions.deleteItems(actionSelectionRefs);
      return;
    }

    if (commandId === 'gallery.starImage' && actionSelectionRefs.length > 0) {
      const loadedItemsByKey = new Map(gallery.items.map((item) => [toGalleryItemKey(item), item]));
      const shouldStar = actionSelectionRefs.some((ref) => !loadedItemsByKey.get(toGalleryItemKey(ref))?.starred);

      void itemActions.setItemsStarred(actionSelectionRefs, shouldStar);
    }
  });

  useEffect(() => {
    const directions = [
      ['gallery.selectAllOnPage', t('widgets.gallery.commands.selectAllOnPage'), null, ['mod+a']],
      ['gallery.clearSelection', t('widgets.gallery.commands.clearSelection'), null, ['esc']],
      ['gallery.galleryNavUp', t('widgets.gallery.commands.navigationUp'), 'up', ['arrowup']],
      ['gallery.galleryNavRight', t('widgets.gallery.commands.navigationRight'), 'right', ['arrowright']],
      ['gallery.galleryNavDown', t('widgets.gallery.commands.navigationDown'), 'down', ['arrowdown']],
      ['gallery.galleryNavLeft', t('widgets.gallery.commands.navigationLeft'), 'left', ['arrowleft']],
      ['gallery.galleryNavUpAlt', t('widgets.gallery.commands.navigationUp'), 'up', ['alt+arrowup']],
      ['gallery.galleryNavRightAlt', t('widgets.gallery.commands.navigationRight'), 'right', ['alt+arrowright']],
      ['gallery.galleryNavDownAlt', t('widgets.gallery.commands.navigationDown'), 'down', ['alt+arrowdown']],
      ['gallery.galleryNavLeftAlt', t('widgets.gallery.commands.navigationLeft'), 'left', ['alt+arrowleft']],
      ['gallery.deleteSelection', t('widgets.gallery.commands.deleteSelection'), null, ['delete', 'backspace']],
      ['gallery.starImage', t('widgets.gallery.commands.toggleStarImage'), null, ['.']],
    ] as const;
    const disposers = directions.flatMap(([id, title, direction, defaultKeys]) => [
      runtime.commands.register({
        handler: () => (direction ? navigate(direction) : executeGalleryHotkey(id)),
        id,
        title,
      }),
      runtime.hotkeys.register({
        commandId: id,
        defaultKeys: [...defaultKeys],
        id,
        title,
      }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys, t]);

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
          <ScrollArea.Viewport ref={viewportRef} h="full" w="full" outline="none">
            <ScrollArea.Content aria-label={t('widgets.gallery.itemsAriaLabel')} role="list">
              <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
                {virtualRows.map((virtualRow) => (
                  <Box
                    key={virtualRow.key}
                    display="grid"
                    gap={`${GRID_GAP_PX}px`}
                    gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
                    left="0"
                    position="absolute"
                    role="presentation"
                    top="0"
                    transform={`translateY(${virtualRow.start}px)`}
                    w="full"
                  >
                    {cells
                      .slice(virtualRow.index * columnCount, (virtualRow.index + 1) * columnCount)
                      .map((cell) =>
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
                            compareRole={
                              isComparisonActive && toGalleryItemKey(cell.item) === gallery.selectedItemKey
                                ? t('widgets.preview.viewing')
                                : isComparisonActive && toGalleryItemKey(cell.item) === gallery.compareImageKey
                                  ? t('widgets.preview.compare')
                                  : null
                            }
                            fit={thumbnailFit}
                            getDragItems={getDragItems}
                            item={cell.item}
                            isPrimary={!isFollowingLive && toGalleryItemKey(cell.item) === gallery.selectedItemKey}
                            isSelected={!isFollowingLive && selectedItemKeys.has(toGalleryItemKey(cell.item))}
                            onClick={handleThumbnailClick}
                            onContextMenu={handleThumbnailContextMenu}
                            onToggleStarred={handleToggleStarred}
                          />
                        )
                      )}
                  </Box>
                ))}
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

const GalleryQueuePlaceholderCell = ({
  antialiasProgressImages,
  fit,
  isSelected,
  onClick,
  placeholder,
}: {
  antialiasProgressImages: boolean;
  fit: GalleryThumbnailFit;
  isSelected: boolean;
  onClick: () => void;
  placeholder: GalleryQueuePlaceholder;
}) => {
  const { t } = useTranslation();
  const progressImage = useQueueItemProgressImage(placeholder.queueItemId, placeholder.itemIndex);
  const progress = useQueueItemProgress(placeholder.queueItemId);
  const isActive = progress?.activeItemIndex === placeholder.itemIndex;
  const percentage = typeof progress?.percentage === 'number' ? Math.round(progress.percentage * 100) : null;

  return (
    <Box aspectRatio={1} minW="0" role="listitem" w="full">
      <Box
        as="button"
        aria-label={t('widgets.preview.showInProgressDiffusion')}
        aria-pressed={isSelected}
        bg="bg"
        borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
        borderWidth={isSelected ? '2px' : '1px'}
        cursor="pointer"
        h="full"
        overflow="hidden"
        position="relative"
        rounded="md"
        w="full"
        onClick={onClick}
      >
        <StreamingImageFrame
          fit={fit === 'aspect' ? 'contain' : 'cover'}
          h="full"
          liveImage={progressImageToStreamingSource(progressImage)}
          shouldAntialiasLiveImage={antialiasProgressImages}
          w="full"
        >
          <Skeleton h="full" w="full" />
        </StreamingImageFrame>
        {isActive ? <GalleryPlaceholderCircularProgress percentage={percentage} /> : null}
      </Box>
    </Box>
  );
};

const GalleryPlaceholderCircularProgress = ({ percentage }: { percentage: number | null }) => {
  const { t } = useTranslation();

  return (
    <Flex
      align="center"
      justify="center"
      alignItems="center"
      pointerEvents="none"
      position="absolute"
      inset="0"
      zIndex="1"
    >
      <ProgressCircle.Root
        aria-label={
          percentage === null
            ? t('widgets.gallery.generationProgress')
            : t('widgets.gallery.generationProgressPercent', { percentage })
        }
        bg="bg/85"
        borderWidth={1}
        p={0.5}
        rounded="full"
        size="xs"
        value={percentage}
      >
        <ProgressCircle.Circle>
          <ProgressCircle.Track />
          <ProgressCircle.Range />
        </ProgressCircle.Circle>
      </ProgressCircle.Root>
    </Flex>
  );
};

const GalleryThumbnail = ({
  alwaysShowDimensions,
  compareRole,
  dragItems,
  fit,
  item,
  isPrimary,
  isSelected,
  onClick,
  onContextMenu,
  onToggleStarred,
}: {
  alwaysShowDimensions: boolean;
  compareRole: string | null;
  dragItems: GalleryItemRef[];
  fit: GalleryThumbnailFit;
  item: GalleryItem;
  isPrimary: boolean;
  isSelected: boolean;
  onClick: (item: GalleryItem, event: MouseEvent) => void;
  onContextMenu: (item: GalleryItem, x: number, y: number) => void;
  onToggleStarred: (item: GalleryItem) => void;
}) => {
  const { t } = useTranslation();
  const isCompared = compareRole !== null;
  const duration = item.kind === 'video' ? formatGalleryVideoDuration(item.durationSeconds) : null;

  const { isDragging, listeners, setNodeRef, transform } = useDraggable({
    data: getGalleryItemDragData(dragItems),
    id: getGalleryItemDragId(toGalleryItemRef(item), 'gallery-grid'),
  });

  const containerStyle = useMemo(() => ({ transform: CSS.Transform.toString(transform) }), [transform]);

  const imageStyle = useMemo(
    () =>
      ({
        display: 'block',
        height: '100%',
        inset: 0,
        maxWidth: 'none',
        objectFit: fit === 'aspect' ? 'contain' : 'cover',
        position: 'absolute',
        width: '100%',
      }) as const,
    [fit]
  );

  const handleContextMenu = useCallback(
    (event: MouseEvent) => {
      event.preventDefault();
      onContextMenu(item, event.clientX, event.clientY);
    },
    [item, onContextMenu]
  );

  const handleClick = useCallback((event: MouseEvent) => onClick(item, event), [item, onClick]);
  const handleActivationKeyDown = useCallback((event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.stopPropagation();
    }
  }, []);

  const handleToggleStarred = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();

      onToggleStarred(item);
    },
    [item, onToggleStarred]
  );

  return (
    <Box
      ref={setNodeRef}
      {...listeners}
      aspectRatio={1}
      bg="bg"
      borderWidth="2px"
      borderColor={isSelected || isCompared ? 'accent.solid' : 'border.subtle'}
      boxShadow={isCompared ? 'inset 0 0 0 1px {colors.accent.solid}' : undefined}
      css={THUMBNAIL_HOVER_CSS}
      minW="0"
      opacity={isDragging ? 0.55 : undefined}
      overflow="hidden"
      role="listitem"
      position="relative"
      rounded="md"
      style={containerStyle}
      touchAction="none"
      w="full"
      zIndex={isDragging ? 2 : undefined}
      onContextMenu={handleContextMenu}
    >
      <button
        aria-current={isPrimary ? 'true' : undefined}
        aria-label={
          item.kind === 'video'
            ? t('widgets.gallery.selectVideoForPreview', { duration, name: item.name })
            : t('widgets.gallery.selectImageForPreview', { name: item.name })
        }
        aria-pressed={isSelected}
        style={THUMBNAIL_BUTTON_STYLE}
        type="button"
        onClick={handleClick}
        onKeyDown={handleActivationKeyDown}
      >
        <img
          alt={item.name}
          decoding={item.kind === 'video' ? 'async' : undefined}
          draggable={false}
          src={item.thumbnailUrl || item.fullUrl}
          style={imageStyle}
        />
      </button>
      {compareRole && (
        <Badge
          insetInlineStart="1"
          pointerEvents="none"
          position="absolute"
          size="xs"
          top="1"
          variant="solid"
          zIndex="1"
        >
          {compareRole}
        </Badge>
      )}
      <IconButton
        aria-label={
          item.starred
            ? t('widgets.gallery.unstarImage', { name: item.name })
            : t('widgets.gallery.starImage', { name: item.name })
        }
        className="gallery-thumb-overlay"
        colorPalette={item.starred ? 'yellow' : 'gray'}
        insetInlineEnd="1"
        opacity={item.starred ? 1 : 0}
        position="absolute"
        size="2xs"
        top="1"
        transition="opacity var(--wb-motion-duration-medium) ease"
        variant="solid"
        zIndex="1"
        onClick={handleToggleStarred}
      >
        <StarIcon fill={item.starred ? 'currentColor' : 'none'} />
      </IconButton>
      {item.kind === 'image' && item.width > 0 && item.height > 0 && (
        <Badge
          bottom="1"
          className="gallery-thumb-overlay"
          insetInlineStart="1"
          opacity={alwaysShowDimensions ? 1 : 0}
          pointerEvents="none"
          position="absolute"
          size="xs"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="solid"
          zIndex="1"
        >
          {item.width}x{item.height}
        </Badge>
      )}
      {duration !== null && (
        <Badge
          bottom="1"
          display="flex"
          fontVariantNumeric="tabular-nums"
          gap="1"
          insetInlineStart="1"
          opacity={1}
          pointerEvents="none"
          position="absolute"
          size="xs"
          transition="opacity var(--wb-motion-duration-medium) ease"
          variant="solid"
          zIndex="1"
        >
          <PlayIcon aria-hidden="true" fill="currentColor" />
          {duration}
        </Badge>
      )}
    </Box>
  );
};

const GalleryThumbnailCell = ({
  item,
  getDragItems,
  ...props
}: Omit<Parameters<typeof GalleryThumbnail>[0], 'dragItems'> & {
  getDragItems: (item: GalleryItem) => GalleryItemRef[];
}) => {
  const dragItems = useMemo(() => getDragItems(item), [getDragItems, item]);

  return <GalleryThumbnail {...props} dragItems={dragItems} item={item} />;
};
