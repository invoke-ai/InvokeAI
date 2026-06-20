import type { GalleryImage } from '@workbench/gallery/api';
import type { GalleryThumbnailFit } from '@workbench/gallery/settings';

import { Badge, Box, Flex, ProgressCircle, ScrollArea, Skeleton, Spinner, Text } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useQueueItemProgressImage } from '@workbench/backend/progressImageStore';
import { useQueueItemProgress } from '@workbench/backend/progressStore';
import { IconButton } from '@workbench/components/ui';
import { ImageContextMenu, type ImageContextMenuTarget } from '@workbench/image-actions';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { StarIcon, UploadIcon } from 'lucide-react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent, type MouseEvent } from 'react';

import type { GalleryImageDragImage } from './galleryDnd';
import type { GalleryQueuePlaceholder } from './galleryStateView';

import { getGalleryImageDragData, getGalleryImageDragId } from './galleryDnd';
import { useGalleryWidget } from './GalleryWidgetContext';

const GRID_GAP_PX = 4;

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
  const { actions, gallery, imageActions, runtime } = useGalleryWidget();
  const dispatch = useWorkbenchDispatch();
  const [contextMenuTarget, setContextMenuTarget] = useState<ImageContextMenuTarget | null>(null);
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(layout) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;
  const antialiasProgressImages = useActiveProjectSelector((project) => project.settings.antialiasProgressImages);
  const columnCount = getGalleryColumnCount(imageDensityPercent, layout);
  const selectedNames = useMemo(() => new Set(gallery.selectedImageNames), [gallery.selectedImageNames]);
  const isComparisonActive =
    gallery.compareImageName !== null &&
    gallery.selectedImageName !== null &&
    gallery.compareImageName !== gallery.selectedImageName;
  const selectedBoardName = gallery.boards.find((board) => board.id === gallery.selectedBoardId)?.name ?? 'this board';
  const isEmpty = gallery.images.length === 0 && gallery.pendingPlaceholders.length === 0;

  // Fresh generations are unstarred, so with newest-first ordering they land
  // right after the starred block — pending placeholders go there too.
  const leadingStarredCount = useMemo(() => {
    if (imageOrderDir !== 'DESC' || !starredFirst) {
      return 0;
    }

    let count = 0;

    while (count < gallery.images.length && gallery.images[count].starred) {
      count += 1;
    }

    return count;
  }, [gallery.images, imageOrderDir, starredFirst]);
  const cells = useMemo<GridCell[]>(() => {
    const imageCells: GridCell[] = gallery.images.map((image, imageIndex) => ({ image, imageIndex, kind: 'image' }));
    const placeholderCells: GridCell[] = gallery.pendingPlaceholders.map((placeholder) => ({
      kind: 'placeholder',
      placeholder,
    }));

    if (imageOrderDir !== 'DESC') {
      return [...imageCells, ...placeholderCells];
    }

    return [...imageCells.slice(0, leadingStarredCount), ...placeholderCells, ...imageCells.slice(leadingStarredCount)];
  }, [gallery.images, gallery.pendingPlaceholders, imageOrderDir, leadingStarredCount]);
  const getCellIndexForImage = (imageIndex: number): number =>
    imageOrderDir === 'DESC' && imageIndex >= leadingStarredCount
      ? imageIndex + gallery.pendingPlaceholders.length
      : imageIndex;
  const rowCount = Math.ceil(cells.length / columnCount);
  const cellSizePx =
    viewportWidth > 0 ? Math.max(1, (viewportWidth - GRID_GAP_PX * (columnCount - 1)) / columnCount) : 96;
  const rowHeightPx = cellSizePx + GRID_GAP_PX;
  const rowHeightRef = useRef(rowHeightPx);

  rowHeightRef.current = rowHeightPx;

  const virtualizer = useVirtualizer({
    count: rowCount,
    estimateSize: () => rowHeightRef.current,
    getScrollElement: () => viewportRef.current,
    overscan: 4,
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
    virtualizer.measure();
  }, [rowHeightPx, virtualizer]);

  const virtualRows = virtualizer.getVirtualItems();
  const lastVisibleRowIndex = virtualRows[virtualRows.length - 1]?.index ?? 0;

  useEffect(() => {
    if (paginationMode === 'infinite' && rowCount > 0 && lastVisibleRowIndex >= rowCount - 2) {
      actions.loadMore();
    }
  }, [actions, lastVisibleRowIndex, paginationMode, rowCount]);

  useEffect(() => {
    const primaryTargetImageName = contextMenuTarget?.images[0]?.imageName;

    if (!primaryTargetImageName) {
      return;
    }

    if (!gallery.images.some((image) => image.imageName === primaryTargetImageName)) {
      setContextMenuTarget(null);
    }
  }, [contextMenuTarget, gallery.images]);

  const handleThumbnailClick = (image: GalleryImage, event: MouseEvent) => {
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
  };

  const handleThumbnailContextMenu = (image: GalleryImage, x: number, y: number) => {
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
  };

  const getDragImages = (image: GalleryImage): GalleryImageDragImage[] => {
    if (selectedNames.has(image.imageName) && selectedNames.size > 1) {
      return gallery.images
        .filter((candidate) => selectedNames.has(candidate.imageName))
        .map((candidate) => ({ boardId: candidate.boardId, imageName: candidate.imageName }));
    }

    return [{ boardId: image.boardId, imageName: image.imageName }];
  };

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
      dispatch({
        type: 'patchWidgetValues',
        values: { selectedImage: null, selectedImageName: null, selectedImageNames: [] },
        widgetId: 'gallery',
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
      ['gallery.selectAllOnPage', 'Select all gallery images on page', null, ['mod+a']],
      ['gallery.clearSelection', 'Clear gallery selection', null, ['esc']],
      ['gallery.galleryNavUp', 'Gallery navigation up', 'up', ['arrowup']],
      ['gallery.galleryNavRight', 'Gallery navigation right', 'right', ['arrowright']],
      ['gallery.galleryNavDown', 'Gallery navigation down', 'down', ['arrowdown']],
      ['gallery.galleryNavLeft', 'Gallery navigation left', 'left', ['arrowleft']],
      ['gallery.galleryNavUpAlt', 'Gallery navigation up', 'up', ['alt+arrowup']],
      ['gallery.galleryNavRightAlt', 'Gallery navigation right', 'right', ['alt+arrowright']],
      ['gallery.galleryNavDownAlt', 'Gallery navigation down', 'down', ['alt+arrowdown']],
      ['gallery.galleryNavLeftAlt', 'Gallery navigation left', 'left', ['alt+arrowleft']],
      ['gallery.deleteSelection', 'Delete gallery selection', null, ['delete', 'backspace']],
      ['gallery.starImage', 'Toggle gallery image star', null, ['.']],
    ] as const;
    const disposers = directions.flatMap(([id, title, direction, defaultKeys]) => [
      runtime.commands.register({
        handler: () => (direction ? navigate(direction) : executeGalleryHotkey(id)),
        id,
        title,
      }),
      runtime.hotkeys.register({ commandId: id, defaultKeys: [...defaultKeys], id, title }),
    ]);

    return () => {
      disposers.forEach((dispose) => dispose());
    };
  }, [runtime.commands, runtime.hotkeys]);

  return (
    <Box
      flex="1"
      maxW="full"
      minH="0"
      minW="0"
      position="relative"
      w="full"
      onDragEnter={(event) => {
        if (!dragEventContainsFiles(event)) {
          return;
        }

        event.preventDefault();
        dragDepthRef.current += 1;
        setIsDropActive(true);
      }}
      onDragLeave={(event) => {
        if (!dragEventContainsFiles(event)) {
          return;
        }

        dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);

        if (dragDepthRef.current === 0) {
          setIsDropActive(false);
        }
      }}
      onDragOver={(event) => {
        if (dragEventContainsFiles(event)) {
          event.preventDefault();
        }
      }}
      onDrop={(event) => {
        event.preventDefault();
        dragDepthRef.current = 0;
        setIsDropActive(false);

        const files = Array.from(event.dataTransfer.files);

        if (files.length > 0) {
          void actions.uploadFiles(files);
        }
      }}
    >
      {isEmpty ? (
        <Flex align="center" color="fg.subtle" h="full" justify="center" minH="8rem">
          <Text fontSize="2xs">
            {gallery.isLoading ? 'Loading backend gallery...' : 'No images match this board, tab, or search.'}
          </Text>
        </Flex>
      ) : (
        <ScrollArea.Root h="full" minH="0" size="xs" variant="hover" w="full">
          <ScrollArea.Viewport
            ref={viewportRef}
            aria-label="Gallery images"
            h="full"
            role="listbox"
            tabIndex={0}
            w="full"
          >
            <ScrollArea.Content>
              <Box h={`${virtualizer.getTotalSize()}px`} position="relative" w="full">
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
                    {cells.slice(virtualRow.index * columnCount, (virtualRow.index + 1) * columnCount).map((cell) =>
                      cell.kind === 'placeholder' ? (
                        <GalleryQueuePlaceholderCell
                          key={cell.placeholder.id}
                          antialiasProgressImages={antialiasProgressImages}
                          fit={thumbnailFit}
                          placeholder={cell.placeholder}
                          onClick={() =>
                            dispatch({
                              settings: { showProgressImagesInViewer: true },
                              type: 'setActiveProjectSettings',
                            })
                          }
                        />
                      ) : (
                        <GalleryThumbnail
                          key={cell.image.imageName}
                          alwaysShowDimensions={showImageDimensions}
                          compareRole={
                            isComparisonActive && cell.image.imageName === gallery.selectedImageName
                              ? 'Viewing'
                              : isComparisonActive && cell.image.imageName === gallery.compareImageName
                                ? 'Compare'
                                : null
                          }
                          fit={thumbnailFit}
                          dragImages={getDragImages(cell.image)}
                          image={cell.image}
                          isPrimary={cell.image.imageName === gallery.selectedImageName}
                          isSelected={selectedNames.has(cell.image.imageName)}
                          onClick={handleThumbnailClick}
                          onContextMenu={handleThumbnailContextMenu}
                          onToggleStarred={(image) =>
                            void imageActions.setImagesStarred([image.imageName], !image.starred)
                          }
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
        <Flex
          align="center"
          bg="bg.muted/80"
          borderColor="accent.solid"
          borderStyle="dashed"
          borderWidth="2px"
          direction="column"
          gap="2"
          inset="0"
          justify="center"
          pointerEvents="none"
          position="absolute"
          rounded="md"
          zIndex="1"
        >
          <UploadIcon size="20" />
          <Text fontSize="xs" fontWeight="600">
            Drop images to upload to {selectedBoardName}
          </Text>
        </Flex>
      )}
      <ImageContextMenu
        actions={imageActions}
        boards={gallery.boards}
        target={contextMenuTarget}
        onClose={() => setContextMenuTarget(null)}
      />
    </Box>
  );
};

const GalleryQueuePlaceholderCell = ({
  antialiasProgressImages,
  fit,
  onClick,
  placeholder,
}: {
  antialiasProgressImages: boolean;
  fit: GalleryThumbnailFit;
  onClick: () => void;
  placeholder: GalleryQueuePlaceholder;
}) => {
  const progressImage = useQueueItemProgressImage(placeholder.queueItemId, placeholder.itemIndex);
  const progress = useQueueItemProgress(placeholder.queueItemId);
  const isActive = progress?.activeItemIndex === placeholder.itemIndex;
  const percentage = typeof progress?.percentage === 'number' ? Math.round(progress.percentage * 100) : null;

  return (
    <Box
      as="button"
      aria-label="Show in-progress diffusion in Preview"
      aspectRatio={1}
      bg="bg"
      borderColor={isActive ? 'accent.solid' : 'border.subtle'}
      borderWidth="1px"
      cursor="pointer"
      minW="0"
      overflow="hidden"
      position="relative"
      rounded="md"
      w="full"
      onClick={onClick}
    >
      {progressImage ? (
        <img
          alt="In-progress diffusion preview"
          draggable={false}
          src={progressImage.dataUrl}
          style={{
            display: 'block',
            height: '100%',
            imageRendering: antialiasProgressImages ? 'auto' : 'pixelated',
            inset: 0,
            maxWidth: 'none',
            objectFit: fit === 'aspect' ? 'contain' : 'cover',
            position: 'absolute',
            width: '100%',
          }}
        />
      ) : (
        <Skeleton h="full" w="full" />
      )}
      {/*<Badge left="1" pointerEvents="none" position="absolute" size="xs" top="1" variant="solid" zIndex="1">
        {isActive ? 'Generating' : 'Queued'}
      </Badge>*/}
      {isActive ? <GalleryPlaceholderCircularProgress percentage={percentage} /> : null}
    </Box>
  );
};

const GalleryPlaceholderCircularProgress = ({ percentage }: { percentage: number | null }) => {
  return (
    <Flex
      align="center"
      aria-label={percentage === null ? 'Generation progress' : `Generation ${percentage}%`}
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
  compareRole: 'Compare' | 'Viewing' | null;
  dragImages: GalleryImageDragImage[];
  fit: GalleryThumbnailFit;
  image: GalleryImage;
  isPrimary: boolean;
  isSelected: boolean;
  onClick: (image: GalleryImage, event: MouseEvent) => void;
  onContextMenu: (image: GalleryImage, x: number, y: number) => void;
  onToggleStarred: (image: GalleryImage) => void;
}) => {
  const isCompared = compareRole !== null;
  const { isDragging, listeners, setNodeRef, transform } = useDraggable({
    data: getGalleryImageDragData(dragImages),
    id: getGalleryImageDragId(image.imageName),
  });

  return (
    <Box
      ref={setNodeRef}
      {...listeners}
      aspectRatio={1}
      bg="bg"
      borderWidth="2px"
      borderColor={isSelected || isCompared ? 'accent.solid' : 'border.subtle'}
      boxShadow={isCompared ? 'inset 0 0 0 1px {colors.accent.solid}' : undefined}
      css={{ '&:hover .gallery-thumb-overlay': { opacity: 1 } }}
      minW="0"
      opacity={isDragging ? 0.55 : undefined}
      overflow="hidden"
      role="option"
      aria-selected={isSelected}
      position="relative"
      rounded="md"
      style={{ transform: CSS.Transform.toString(transform) }}
      touchAction="none"
      w="full"
      zIndex={isDragging ? 2 : undefined}
      onContextMenu={(event) => {
        event.preventDefault();
        onContextMenu(image, event.clientX, event.clientY);
      }}
    >
      <button
        aria-label={`Select ${image.imageName} for preview`}
        style={{
          background: 'transparent',
          border: 0,
          cursor: 'pointer',
          display: 'block',
          height: '100%',
          inset: 0,
          minWidth: 0,
          padding: 0,
          position: 'absolute',
          width: '100%',
        }}
        tabIndex={isPrimary ? 0 : -1}
        type="button"
        onClick={(event) => onClick(image, event)}
      >
        <img
          alt={image.imageName}
          draggable={false}
          src={image.thumbnailUrl || image.imageUrl}
          style={{
            display: 'block',
            height: '100%',
            inset: 0,
            maxWidth: 'none',
            objectFit: fit === 'aspect' ? 'contain' : 'cover',
            position: 'absolute',
            width: '100%',
          }}
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
        aria-label={image.starred ? `Unstar ${image.imageName}` : `Star ${image.imageName}`}
        className="gallery-thumb-overlay"
        colorPalette={image.starred ? 'yellow' : 'gray'}
        insetInlineEnd="1"
        opacity={image.starred ? 1 : 0}
        position="absolute"
        size="2xs"
        top="1"
        transition="opacity 0.15s ease"
        variant="solid"
        zIndex="1"
        onClick={(event) => {
          event.stopPropagation();
          onToggleStarred(image);
        }}
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
          transition="opacity 0.15s ease"
          variant="solid"
          zIndex="1"
        >
          {image.width}x{image.height}
        </Badge>
      )}
    </Box>
  );
};
