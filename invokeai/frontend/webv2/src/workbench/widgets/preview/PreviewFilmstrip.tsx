import type { GalleryItem, GalleryItemKey } from '@features/gallery';

import { Box, HStack, ScrollArea } from '@chakra-ui/react';
import { useDraggable } from '@dnd-kit/core';
import { toGalleryItemKey, toGalleryItemRef } from '@features/gallery/contracts';
import { getGalleryItemDragData, getGalleryItemDragId } from '@features/gallery/utility';
import { useCallback, useMemo } from 'react';

import type { PreviewDensity } from './previewDensity';

/**
 * A docked strip of the current board's thumbnails between frame and footer —
 * the same `boardImages` the "N of M" counter is derived from, made spatial.
 * Fixed height (`flexShrink=0`) so the fitted frame's `100cqh` math is never
 * stolen from. Thumbs are standard all-image gallery-item drag sources, so they work
 * with every existing drop target (canvas zones, boards, drop-to-compare).
 */

export const PreviewFilmstrip = ({
  density,
  items,
  selectedItemKey,
  onSelect,
}: {
  density: PreviewDensity;
  items: GalleryItem[];
  selectedItemKey: GalleryItemKey | null;
  onSelect: (item: GalleryItem) => void;
}) => {
  const thumbSize = density === 'full' ? '12' : '8';

  if (items.length < 2) {
    return null;
  }

  return (
    // Two containment rules keep the strip honest: the ScrollArea root's
    // recipe defaults to `height: 100%`, so it MUST get an explicit height or
    // it swallows the widget; and `contain: inline-size` zeroes the strip's
    // intrinsic width so a long board can never stretch the widget wider than
    // its panel (side panels host widgets in a grid ScrollArea.Content that
    // otherwise grows to max-content).
    <ScrollArea.Root
      css={FILMSTRIP_CONTAIN_CSS}
      flexShrink={0}
      h={density === 'full' ? '3.75rem' : '2.75rem'}
      minW="0"
      size="xs"
      variant="hover"
      w="full"
    >
      <ScrollArea.Viewport h="full" w="full">
        <ScrollArea.Content asChild>
          <HStack align="center" gap="1" h="full">
            {items.map((item) => {
              const itemKey = toGalleryItemKey(item);

              return (
                <FilmstripThumb
                  key={itemKey}
                  item={item}
                  isSelected={itemKey === selectedItemKey}
                  size={thumbSize}
                  onSelect={onSelect}
                />
              );
            })}
          </HStack>
        </ScrollArea.Content>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar orientation="horizontal">
        <ScrollArea.Thumb />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  );
};

const FilmstripThumb = ({
  item,
  isSelected,
  size,
  onSelect,
}: {
  item: GalleryItem;
  isSelected: boolean;
  size: string;
  onSelect: (item: GalleryItem) => void;
}) => {
  const itemRef = useMemo(() => toGalleryItemRef(item), [item]);
  const dragData = useMemo(() => getGalleryItemDragData([itemRef]), [itemRef]);
  const thumbnailSrc = item.thumbnailUrl || (item.kind === 'image' ? item.fullUrl : null);
  const { listeners, setNodeRef } = useDraggable({
    data: dragData,
    id: getGalleryItemDragId(itemRef, 'preview-filmstrip'),
  });
  const handleClick = useCallback(() => onSelect(item), [item, onSelect]);
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
      aria-label={item.kind === 'video' ? `Video ${item.name}` : item.name}
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
      {thumbnailSrc ? (
        <img alt={item.name} loading="lazy" src={thumbnailSrc} style={FILMSTRIP_IMG_STYLE} />
      ) : (
        <Box aria-hidden bg="bg.muted" h="full" w="full" />
      )}
      {isSelected ? <Box bg="accent.solid" bottom="0" h="2px" left="0" position="absolute" right="0" /> : null}
    </Box>
  );
};

const FILMSTRIP_CONTAIN_CSS = { contain: 'inline-size' } as const;

const FILMSTRIP_IMG_STYLE = {
  display: 'block',
  height: '100%',
  objectFit: 'cover',
  width: '100%',
} as const;
