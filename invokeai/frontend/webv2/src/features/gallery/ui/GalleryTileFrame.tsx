import type { BoxProps, SystemStyleObject } from '@chakra-ui/react';
import type { GalleryItem } from '@features/gallery/core/items';
import type { GalleryThumbnailFit } from '@features/gallery/core/settings';
import type { ReactNode, Ref } from 'react';

import { Badge, Box } from '@chakra-ui/react';
import { formatGalleryVideoDuration } from '@features/gallery/core/items';
import { PlayIcon } from 'lucide-react';
import { useMemo } from 'react';

/**
 * Hover tints the whole square cell; the media box keeps the border, selection and focus ring. Overlays stay on the
 * cell's corners, on solid badge fills that read over the tint as well as over the image.
 */
const TILE_CSS: SystemStyleObject = {
  _hover: { bg: 'bg.hover' },
  '&:hover .gallery-thumb-overlay, &:focus-within .gallery-thumb-overlay': { opacity: 1 },
};

/** The box drawn around the media's own rect, inside the square cell. */
export const GALLERY_TILE_MEDIA_SELECTOR = '[data-gallery-tile-media]';

/** Panoramas and strips stop here and cover-crop, so a dense cell never shrinks them to a sliver. */
const MAX_MEDIA_RATIO = 3;

/**
 * Aspect fit sizes the media box to the media's own ratio, clamped, when its dimensions are known; otherwise the box
 * fills the square cell and the media letterboxes inside it.
 */
export const getGalleryTileMediaRatio = (
  { height, width }: { height: number; width: number },
  fit: GalleryThumbnailFit
): number | null =>
  fit === 'aspect' && width > 0 && height > 0
    ? Math.min(MAX_MEDIA_RATIO, Math.max(1 / MAX_MEDIA_RATIO, width / height))
    : null;

/** An inner ring over the image thickens the selected edge without resizing the tile content. */
const SELECTED_TILE_CSS: SystemStyleObject = {
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

const MEDIA_CSS: SystemStyleObject = {
  '&:focus-within': { outline: '2px solid {colors.accent.solid}', outlineOffset: '-2px' },
};
const SELECTED_MEDIA_CSS: SystemStyleObject = { ...MEDIA_CSS, ...SELECTED_TILE_CSS };

export interface GalleryTileMediaProps {
  children?: ReactNode;
  fit: GalleryThumbnailFit;
  height: number;
  isSelected: boolean;
  width: number;
}

/**
 * The bordered box around the media's own rect, inside a tile's square cell; it carries the border, selection ring
 * and focus ring. Centred in the flex cell, its width alone fixes its rect: a portrait box narrows to the ratio and
 * its aspect-ratio brings the height back to the cell's, so it is sized before the media loads, without a jump.
 */
export const GalleryTileMedia = ({ children, fit, height, isSelected, width }: GalleryTileMediaProps) => {
  const ratio = getGalleryTileMediaRatio({ height, width }, fit) ?? 1;
  // Inline: per-item ratios would otherwise each add stylesheet rules on the virtualized grid's hot path.
  const style = useMemo(
    () => ({ aspectRatio: String(ratio), width: ratio >= 1 ? '100%' : `${String(ratio * 100)}%` }),
    [ratio]
  );

  return (
    <Box
      bg="bg"
      borderColor={isSelected ? 'accent.solid' : 'border.subtle'}
      borderWidth="2px"
      css={isSelected ? SELECTED_MEDIA_CSS : MEDIA_CSS}
      data-gallery-tile-media=""
      flexShrink={0}
      overflow="hidden"
      position="relative"
      rounded="md"
      style={style}
    >
      {children}
    </Box>
  );
};

const BADGE_TRANSITION = 'opacity var(--wb-motion-duration-medium) ease';

export interface GalleryTileFrameProps extends Omit<BoxProps, 'children'> {
  alwaysShowDimensions?: boolean;
  /** A `GalleryTileMedia` with the image, then overlays (badges, star toggle) on the square cell's corners. */
  children?: ReactNode;
  item: GalleryItem;
  ref?: Ref<HTMLDivElement>;
}

/** Caller CSS composes with shared tile styling; shell props remain invariant. */
export const GalleryTileFrame = ({
  alwaysShowDimensions = false,
  children,
  css,
  item,
  ...boxProps
}: GalleryTileFrameProps) => {
  const duration = item.kind === 'video' ? formatGalleryVideoDuration(item.durationSeconds) : null;
  const tileCss = useMemo(() => [TILE_CSS, css].filter((entry) => entry !== undefined), [css]);

  return (
    <Box
      {...boxProps}
      alignItems="center"
      aspectRatio={1}
      css={tileCss}
      display="flex"
      justifyContent="center"
      minW="0"
      position="relative"
      rounded="md"
      transition="background-color var(--wb-motion-duration-fast) ease"
      // A shift-click beside the image extends the selection; it must not select badge or nearby text.
      userSelect="none"
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
