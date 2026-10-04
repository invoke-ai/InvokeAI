import type { BoxProps, SystemStyleObject } from '@chakra-ui/react';
import type { GalleryItem } from '@features/gallery/core/items';
import type { ReactNode, Ref } from 'react';

import { Badge, Box } from '@chakra-ui/react';
import { formatGalleryVideoDuration } from '@features/gallery/core/items';
import { PlayIcon } from 'lucide-react';
import { useMemo } from 'react';

const TILE_CSS: SystemStyleObject = {
  '&:focus-within': { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' },
  '&:hover .gallery-thumb-overlay, &:focus-within .gallery-thumb-overlay': { opacity: 1 },
};

/** An inner ring over the image thickens the selected edge without resizing the tile content; shared by every tile. */
export const SELECTED_TILE_CSS: SystemStyleObject = {
  _after: {
    borderRadius: 'sm',
    boxShadow: 'inset 0 0 0 2px {colors.accent.solid}',
    content: '""',
    inset: 0,
    pointerEvents: 'none',
    position: 'absolute',
    zIndex: 1,
  },
};

const BADGE_TRANSITION = 'opacity var(--wb-motion-duration-medium) ease';

export interface GalleryTileFrameProps extends Omit<BoxProps, 'children'> {
  alwaysShowDimensions?: boolean;
  /** The interactive layer (image button, drag listeners, extra badges). */
  children?: ReactNode;
  isSelected?: boolean;
  item: GalleryItem;
  ref?: Ref<HTMLDivElement>;
}

/** Caller CSS composes with shared tile styling; shell props remain invariant. */
export const GalleryTileFrame = ({
  alwaysShowDimensions = false,
  children,
  css,
  isSelected = false,
  item,
  ...boxProps
}: GalleryTileFrameProps) => {
  const duration = item.kind === 'video' ? formatGalleryVideoDuration(item.durationSeconds) : null;
  const tileCss = useMemo(
    () => [TILE_CSS, isSelected ? SELECTED_TILE_CSS : undefined, css].filter((entry) => entry !== undefined),
    [css, isSelected]
  );

  return (
    <Box
      {...boxProps}
      aspectRatio={1}
      bg="bg"
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      css={tileCss}
      minW="0"
      overflow="hidden"
      position="relative"
      rounded="md"
      w="full"
    >
      {children}
      {item.kind === 'image' && item.width > 0 && item.height > 0 ? (
        <Badge
          bottom="1"
          className="gallery-thumb-overlay"
          insetInlineStart="1"
          opacity={alwaysShowDimensions ? 1 : 0}
          pointerEvents="none"
          position="absolute"
          transition={BADGE_TRANSITION}
          variant="solid"
          zIndex="1"
        >
          {item.width}x{item.height}
        </Badge>
      ) : null}
      {duration !== null ? (
        <Badge
          bottom="1"
          display="flex"
          fontVariantNumeric="tabular-nums"
          gap="1"
          insetInlineStart="1"
          pointerEvents="none"
          position="absolute"
          variant="solid"
          zIndex="1"
        >
          <PlayIcon aria-hidden="true" fill="currentColor" />
          {duration}
        </Badge>
      ) : null}
    </Box>
  );
};
