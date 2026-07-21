import type { GalleryThumbnailFit } from '@features/gallery/core/settings';
import type { GalleryImage } from '@features/gallery/core/types';

import { Badge, Box, Flex, ProgressCircle, ScrollArea, Skeleton, Spinner, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { useQueueItemProgress, useQueueItemProgressImage } from '@features/queue/react';
import { DropZone, IconButton } from '@platform/ui';
import { StreamingImageFrame } from '@platform/ui/streaming-image/StreamingImageFrame';
import { progressImageToStreamingSource } from '@platform/ui/streaming-image/streamingImageSource';
import { StarIcon, UploadIcon } from 'lucide-react';
import {
  useEffect,
  useEffectEvent,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type MouseEvent,
} from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { GalleryImageDragImage } from './galleryDnd';
import type { GalleryQueuePlaceholder } from './galleryStateView';
import type { GalleryImageContextMenuTarget } from './GalleryUiContext';

import { getGalleryImageDragData, getGalleryImageDragId } from './galleryDnd';
import { getGalleryPlaceholderInsertionIndex } from './galleryStateView';
import { useGalleryUi } from './GalleryUiContext';
import { useGalleryWidget } from './GalleryWidgetContext';

const GRID_GAP_PX = 4;
const THUMBNAIL_HOVER_CSS = {
  '&:hover .gallery-thumb-overlay': { opacity: 1 },
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

type GridCell =
  | { kind: 'image'; image: GalleryImage; imageIndex: number }
  | { kind: 'placeholder'; placeholder: GalleryQueuePlaceholder };

export const GalleryImageGrid = ({ layout }: { layout: 'stacked' | 'wide' }) => {
  const { t } = useTranslation();
  const { actions, gallery, imageActions, runtime } = useGalleryWidget();
  const { account, antialiasProgressImages, ImageContextMenu, widgets } = useGalleryUi();
  const [contextMenuTarget, setContextMenuTarget] = useState<GalleryImageContextMenuTarget | null>(null);
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(layout) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;
  const columnCount = getGalleryColumnCount(imageDensityPercent, layout);
  const selectedNames = useMemo(() => new Set(gallery.selectedImageNames), [gallery.selectedImageNames]);
  const isFollowingLive = gallery.currentItem?.kind === 'placeholder';
  const isComparisonActive =
    gallery.compareImageName !== null &&
    gallery.selectedImageName !== null &&
    gallery.compareImageName !== gallery.selectedImageName;
  const selectedBoardName =
    gallery.boards.find((board) => board.id === gallery.selectedBoardId)?.name ??
    t('widgets.gallery.selectedBoardFallback');
  const isEmpty = gallery.images.length === 0 && gallery.pendingPlaceholders.length === 0;

  const placeholderInsertionIndex = getGalleryPlaceholderInsertionIndex(gallery.images, imageOrderDir, starredFirst);

  const cells = useMemo<GridCell[]>(() => {
    const imageCells: GridCell[] = gallery.images.map((image, imageIndex) => ({
      image,
      imageIndex,
      kind: 'image',
    }));
    const placeholderCells: GridCell[] = gallery.pendingPlaceholders.map((placeholder) => ({
      kind: 'placeholder',
      placeholder,
    }));

    return [
      ...imageCells.slice(0, placeholderInsertionIndex),
      ...placeholderCells,
      ...imageCells.slice(placeholderInsertionIndex),
    ];
  }, [gallery.images, gallery.pendingPlaceholders, placeholderInsertionIndex]);

  const getCellIndexForImage = (imageIndex: number): number =>
    imageIndex >= placeholderInsertionIndex ? imageIndex + gallery.pendingPlaceholders.length : imageIndex;

  const rowCount = Math.ceil(cells.length / columnCount);

  const cellSizePx =
    viewportWidth > 0 ? Math.max(1, (viewportWidth - GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;

  const rowHeightPx = cellSizePx + GRID_GAP_PX;

  const virtualizer = useVirtualizer({
    count: rowCount,
    estimateSize: () => rowHeightPx,
    getScrollElement: () => viewportRef.current,
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

  const activeContextMenuTarget = useMemo(() => {
    const primaryTargetImageName = contextMenuTarget?.images[0]?.imageName;

    if (!primaryTargetImageName) {
      return contextMenuTarget;
    }

    return gallery.images.some((image) => image.imageName === primaryTargetImageName) ? contextMenuTarget : null;
  }, [contextMenuTarget, gallery.images]);

  const handleThumbnailClick = useCallback(
    (image: GalleryImage, event: MouseEvent) => {
      if (event.shiftKey) {
        const targetIndex = gallery.images.findIndex((candidate) => candidate.imageName === image.imageName);
        const anchorIndex = gallery.images.findIndex((candidate) => candidate.imageName === gallery.selectedImageName);

        if (targetIndex !== -1 && anchorIndex !== -1 && targetIndex !== anchorIndex) {
          const start = Math.min(anchorIndex, targetIndex);
          const end = Math.max(anchorIndex, targetIndex);

          actions.selectImageRange(
            gallery.images.slice(start, end + 1).map((candidate) => candidate.imageName),
            image
          );
          return;
        }
      }

      if (event.ctrlKey || event.metaKey) {
        actions.toggleImageInSelection(image);
      } else {
        actions.selectImage(image);
      }
    },
    [actions, gallery.images, gallery.selectedImageName]
  );

  const handleThumbnailContextMenu = useCallback(
    (image: GalleryImage, x: number, y: number) => {
      if (selectedNames.has(image.imageName) && selectedNames.size > 1) {
        const selectionImages = [
          image,
          ...gallery.images.filter(
            (candidate) => candidate.imageName !== image.imageName && selectedNames.has(candidate.imageName)
          ),
        ];

        setContextMenuTarget({ images: selectionImages, x, y });
        return;
      }

      setContextMenuTarget({ images: [image], x, y });
    },
    [gallery.images, selectedNames]
  );

  const getDragImages = useCallback(
    (image: GalleryImage): GalleryImageDragImage[] => {
      if (selectedNames.has(image.imageName) && selectedNames.size > 1) {
        return gallery.images
          .filter((candidate) => selectedNames.has(candidate.imageName))
          .map((candidate) => ({
            boardId: candidate.boardId,
            imageName: candidate.imageName,
          }));
      }

      return [{ boardId: image.boardId, imageName: image.imageName }];
    },
    [gallery.images, selectedNames]
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
    (image: GalleryImage) => void imageActions.setImagesStarred([image.imageName], !image.starred),
    [imageActions]
  );

  const handleCloseContextMenu = useCallback(() => setContextMenuTarget(null), []);

  const navigate = useEffectEvent((direction: 'down' | 'left' | 'right' | 'up') => {
    const imageCount = gallery.images.length;

    if (imageCount === 0) {
      return;
    }

    const selectedIndex = gallery.images.findIndex((image) => image.imageName === gallery.selectedImageName);
    const fallbackIndex = selectedIndex === -1 ? 0 : selectedIndex;
    const delta =
      direction === 'right' ? 1 : direction === 'left' ? -1 : direction === 'down' ? columnCount : -columnCount;
    const nextIndex = Math.min(imageCount - 1, Math.max(0, fallbackIndex + delta));
    const nextImage = gallery.images[nextIndex];

    if (nextImage && (nextIndex !== selectedIndex || selectedIndex === -1)) {
      actions.selectImage(nextImage);
      virtualizer.scrollToIndex(Math.floor(getCellIndexForImage(nextIndex) / columnCount));
    }
  });

  const executeGalleryHotkey = useEffectEvent((commandId: string) => {
    if (commandId === 'gallery.selectAllOnPage') {
      const primaryImage = gallery.images[0];

      if (primaryImage) {
        actions.selectImageRange(
          gallery.images.map((image) => image.imageName),
          primaryImage
        );
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

    if (commandId === 'gallery.deleteSelection' && gallery.selectedImageNames.length > 0) {
      void imageActions.deleteImages(gallery.selectedImageNames);
      return;
    }

    if (commandId === 'gallery.starImage') {
      const imageNames = gallery.selectedImageNames.length
        ? gallery.selectedImageNames
        : gallery.selectedImageName
          ? [gallery.selectedImageName]
          : [];
      const selectedImages = gallery.images.filter((image) => imageNames.includes(image.imageName));

      if (selectedImages.length > 0) {
        const shouldStar = selectedImages.some((image) => !image.starred);

        void imageActions.setImagesStarred(imageNames, shouldStar);
      }
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
        <Flex align="center" color="fg.subtle" h="full" justify="center" minH="8rem">
          <Text fontSize="2xs">
            {gallery.isLoading ? t('widgets.gallery.loadingBackendGallery') : t('widgets.gallery.noImagesMatch')}
          </Text>
        </Flex>
      ) : (
        <ScrollArea.Root h="full" minH="0" size="xs" variant="hover" w="full">
          <ScrollArea.Viewport
            ref={viewportRef}
            aria-label={t('widgets.gallery.imagesAriaLabel')}
            h="full"
            role="listbox"
            tabIndex={0}
            w="full"
            outline="none"
          >
            <ScrollArea.Content>
              <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
                {virtualRows.map((virtualRow) => (
                  <Box
                    key={virtualRow.key}
                    display="grid"
                    gap={`${GRID_GAP_PX}px`}
                    gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
                    left="0"
                    position="absolute"
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
                            key={cell.image.imageName}
                            alwaysShowDimensions={showImageDimensions}
                            compareRole={
                              isComparisonActive && cell.image.imageName === gallery.selectedImageName
                                ? t('widgets.preview.viewing')
                                : isComparisonActive && cell.image.imageName === gallery.compareImageName
                                  ? t('widgets.preview.compare')
                                  : null
                            }
                            fit={thumbnailFit}
                            getDragImages={getDragImages}
                            image={cell.image}
                            isPrimary={!isFollowingLive && cell.image.imageName === gallery.selectedImageName}
                            isSelected={!isFollowingLive && selectedNames.has(cell.image.imageName)}
                            onClick={handleThumbnailClick}
                            onContextMenu={handleThumbnailContextMenu}
                            onToggleStarred={handleToggleStarred}
                          />
                        )
                      )}
                  </Box>
                ))}
              </Box>
              {paginationMode === 'infinite' && gallery.isLoading && gallery.images.length > 0 && (
                <Flex align="center" justify="center" py="2">
                  <Spinner color="fg.subtle" size="xs" />
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
            {t('widgets.gallery.dropImagesToUploadToBoard', { name: selectedBoardName })}
          </Text>
        </DropZone>
      )}
      <ImageContextMenu
        actions={imageActions}
        boards={gallery.boards}
        target={activeContextMenuTarget}
        onClose={handleCloseContextMenu}
      />
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
    <Box
      as="button"
      aria-label={t('widgets.preview.showInProgressDiffusion')}
      aria-selected={isSelected}
      aspectRatio={1}
      bg="bg"
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth={isSelected ? '2px' : '1px'}
      cursor="pointer"
      minW="0"
      rounded="md"
      overflow="hidden"
      position="relative"
      role="option"
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
  );
};

const GalleryPlaceholderCircularProgress = ({ percentage }: { percentage: number | null }) => {
  const { t } = useTranslation();

  return (
    <Flex
      align="center"
      aria-label={
        percentage === null
          ? t('widgets.gallery.generationProgress')
          : t('widgets.gallery.generationProgressPercent', { percentage })
      }
      justify="center"
      alignItems="center"
      pointerEvents="none"
      position="absolute"
      inset="0"
      zIndex="1"
    >
      <ProgressCircle.Root value={percentage} size="xs" bg="bg/85" rounded="full" borderWidth={1} p={0.5}>
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
  dragImages,
  fit,
  image,
  isPrimary,
  isSelected,
  onClick,
  onContextMenu,
  onToggleStarred,
}: {
  alwaysShowDimensions: boolean;
  compareRole: string | null;
  dragImages: GalleryImageDragImage[];
  fit: GalleryThumbnailFit;
  image: GalleryImage;
  isPrimary: boolean;
  isSelected: boolean;
  onClick: (image: GalleryImage, event: MouseEvent) => void;
  onContextMenu: (image: GalleryImage, x: number, y: number) => void;
  onToggleStarred: (image: GalleryImage) => void;
}) => {
  const { t } = useTranslation();
  const isCompared = compareRole !== null;

  const { isDragging, listeners, setNodeRef, transform } = useDraggable({
    data: getGalleryImageDragData(dragImages),
    id: getGalleryImageDragId(image.imageName),
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
      onContextMenu(image, event.clientX, event.clientY);
    },
    [image, onContextMenu]
  );

  const handleClick = useCallback((event: MouseEvent) => onClick(image, event), [image, onClick]);

  const handleToggleStarred = useCallback(
    (event: MouseEvent<HTMLButtonElement>) => {
      event.stopPropagation();
      onToggleStarred(image);
    },
    [image, onToggleStarred]
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
      role="option"
      aria-selected={isSelected}
      position="relative"
      rounded="md"
      style={containerStyle}
      touchAction="none"
      w="full"
      zIndex={isDragging ? 2 : undefined}
      onContextMenu={handleContextMenu}
    >
      <button
        aria-label={t('widgets.gallery.selectImageForPreview', { name: image.imageName })}
        style={THUMBNAIL_BUTTON_STYLE}
        tabIndex={isPrimary ? 0 : -1}
        type="button"
        onClick={handleClick}
      >
        <img alt={image.imageName} draggable={false} src={image.thumbnailUrl || image.imageUrl} style={imageStyle} />
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
          image.starred
            ? t('widgets.gallery.unstarImage', { name: image.imageName })
            : t('widgets.gallery.starImage', { name: image.imageName })
        }
        className="gallery-thumb-overlay"
        colorPalette={image.starred ? 'yellow' : 'gray'}
        insetInlineEnd="1"
        opacity={image.starred ? 1 : 0}
        position="absolute"
        size="2xs"
        top="1"
        transition="opacity var(--wb-motion-duration-medium) ease"
        variant="solid"
        zIndex="1"
        onClick={handleToggleStarred}
      >
        <StarIcon fill={image.starred ? 'currentColor' : 'none'} />
      </IconButton>
      {image.width > 0 && image.height > 0 && (
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
          {image.width}x{image.height}
        </Badge>
      )}
    </Box>
  );
};

const GalleryThumbnailCell = ({
  image,
  getDragImages,
  ...props
}: Omit<Parameters<typeof GalleryThumbnail>[0], 'dragImages'> & {
  getDragImages: (image: GalleryImage) => GalleryImageDragImage[];
}) => {
  const dragImages = useMemo(() => getDragImages(image), [getDragImages, image]);

  return <GalleryThumbnail {...props} dragImages={dragImages} image={image} />;
};
