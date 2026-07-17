import type { GeneratedImageContract } from '@workbench/types';

import { Box, HStack } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { getGalleryImageDragData } from '@workbench/widgets/gallery/galleryDnd';
import { useCallback, useMemo } from 'react';

import type { PreviewDensity } from './previewDensity';

/**
 * A docked strip of the current board's thumbnails between frame and footer —
 * the same `boardImages` the "N of M" counter is derived from, made spatial.
 * Fixed height (`flexShrink=0`) so the fitted frame's `100cqh` math is never
 * stolen from. Thumbs are standard gallery-image drag sources, so they work
 * with every existing drop target (canvas zones, boards, drop-to-compare).
 */

type FilmstripImage = GeneratedImageContract & { boardId?: string };

export const PreviewFilmstrip = ({
  density,
  images,
  selectedImageName,
  onSelect,
}: {
  density: PreviewDensity;
  images: FilmstripImage[];
  selectedImageName: string | null;
  onSelect: (image: GeneratedImageContract) => void;
}) => {
  const thumbSize = density === 'full' ? '12' : '8';

  if (images.length < 2) {
    return null;
  }

  return (
    <HStack flexShrink={0} gap="1" overflowX="auto" overflowY="hidden" pb="1" w="full">
      {images.map((image) => (
        <FilmstripThumb
          key={image.imageName}
          image={image}
          isSelected={image.imageName === selectedImageName}
          size={thumbSize}
          onSelect={onSelect}
        />
      ))}
    </HStack>
  );
};

const FilmstripThumb = ({
  image,
  isSelected,
  size,
  onSelect,
}: {
  image: FilmstripImage;
  isSelected: boolean;
  size: string;
  onSelect: (image: GeneratedImageContract) => void;
}) => {
  const dragData = useMemo(
    () => getGalleryImageDragData([{ boardId: image.boardId ?? 'none', imageName: image.imageName }]),
    [image.boardId, image.imageName]
  );
  const { listeners, setNodeRef } = useDraggable({
    data: dragData,
    id: `preview-filmstrip:${image.imageName}`,
  });
  const handleClick = useCallback(() => onSelect(image), [image, onSelect]);
  // Ref callbacks re-run when `isSelected` changes, keeping the selected thumb
  // in view without an effect.
  const scrollIntoView = useCallback(
    (node: HTMLElement | null) => {
      setNodeRef(node);

      if (node && isSelected) {
        node.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    },
    [isSelected, setNodeRef]
  );

  return (
    <Box
      ref={scrollIntoView}
      {...listeners}
      as="button"
      aria-current={isSelected || undefined}
      aria-label={image.imageName}
      borderColor={isSelected ? 'border.emphasized' : 'border.subtle'}
      borderWidth="1px"
      boxSize={size}
      cursor="pointer"
      flexShrink={0}
      overflow="hidden"
      position="relative"
      rounded="sm"
      touchAction="none"
      onClick={handleClick}
    >
      <img
        alt={image.imageName}
        loading="lazy"
        src={image.thumbnailUrl || image.imageUrl}
        style={FILMSTRIP_IMG_STYLE}
      />
      {isSelected ? <Box bg="accent.solid" bottom="0" h="2px" left="0" position="absolute" right="0" /> : null}
    </Box>
  );
};

const FILMSTRIP_IMG_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'cover',
  width: '100%',
} as const;
