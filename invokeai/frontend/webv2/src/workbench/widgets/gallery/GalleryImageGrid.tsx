import { Badge, Box, Flex, ScrollArea, Skeleton, Spinner, Text } from '@chakra-ui/react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent, type MouseEvent } from 'react';
import { StarIcon, UploadIcon } from 'lucide-react';

import { ImageContextMenu, type ImageContextMenuTarget } from '../../components/ImageContextMenu';
import { IconButton } from '../../components/ui/Button';
import type { GalleryImage } from '../../gallery/api';
import type { GalleryThumbnailFit } from '../../gallery/settings';
import type { GalleryQueuePlaceholder } from './galleryStateView';
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
  const { actions, gallery, imageActions } = useGalleryWidget();
  const [contextMenuTarget, setContextMenuTarget] = useState<ImageContextMenuTarget | null>(null);
  const [isDropActive, setIsDropActive] = useState(false);
  const [viewportWidth, setViewportWidth] = useState(() => viewportWidthCache.get(layout) ?? 0);
  const dragDepthRef = useRef(0);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { imageDensityPercent, imageOrderDir, paginationMode, showImageDimensions, starredFirst, thumbnailFit } =
    gallery.settings;
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

  const navigate = (direction: 'down' | 'left' | 'right' | 'up') => {
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
  };

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
            onKeyDown={(event) => {
              const direction =
                event.key === 'ArrowRight'
                  ? ('right' as const)
                  : event.key === 'ArrowLeft'
                    ? ('left' as const)
                    : event.key === 'ArrowDown'
                      ? ('down' as const)
                      : event.key === 'ArrowUp'
                        ? ('up' as const)
                        : null;

              if (direction) {
                event.preventDefault();
                navigate(direction);
              }
            }}
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
                    {cells
                      .slice(virtualRow.index * columnCount, (virtualRow.index + 1) * columnCount)
                      .map((cell) =>
                        cell.kind === 'placeholder' ? (
                          <Skeleton key={cell.placeholder.id} aspectRatio={1} minW="0" rounded="md" w="full" />
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
          bg="bg.surfaceRaised/80"
          borderColor="accent.active"
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

const GalleryThumbnail = ({
  alwaysShowDimensions,
  compareRole,
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
  fit: GalleryThumbnailFit;
  image: GalleryImage;
  isPrimary: boolean;
  isSelected: boolean;
  onClick: (image: GalleryImage, event: MouseEvent) => void;
  onContextMenu: (image: GalleryImage, x: number, y: number) => void;
  onToggleStarred: (image: GalleryImage) => void;
}) => {
  const isCompared = compareRole !== null;

  return (
    <Box
      aspectRatio={1}
      bg="bg.panel"
      borderWidth="2px"
      borderColor={isSelected || isCompared ? 'accent.active' : 'border.subtle'}
      boxShadow={isCompared ? 'inset 0 0 0 1px var(--chakra-colors-accent-active)' : undefined}
      css={{ '&:hover .gallery-thumb-overlay': { opacity: 1 } }}
      minW="0"
      overflow="hidden"
      role="option"
      aria-selected={isSelected}
      position="relative"
      rounded="md"
      w="full"
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
