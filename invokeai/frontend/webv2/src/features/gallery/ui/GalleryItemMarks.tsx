import type { GalleryColorLabel, GalleryItemMarkFields } from '@features/gallery/core/marks';
import type { TFunction } from 'i18next';

import { Badge, Box, HStack, Icon, type BoxProps, type IconProps } from '@chakra-ui/react';
import { BookmarkIcon, FlagIcon, StarIcon } from 'lucide-react';

/** The theme's categorical palettes; yellow stays the rating stars' own. */
export const GALLERY_COLOR_LABEL_PALETTES: Record<GalleryColorLabel, string> = {
  blue: 'azure',
  green: 'mint',
  purple: 'lavender',
  red: 'coral',
  yellow: 'gold',
};

export const getGalleryColorLabelName = (t: TFunction, colorLabel: GalleryColorLabel): string =>
  t(`widgets.gallery.marks.colorLabels.${colorLabel}`);

/** "Bookmarked, 3 stars, Red label"; null for an unmarked item. */
export const getGalleryItemMarksDescription = (t: TFunction, item: GalleryItemMarkFields): string | null => {
  const parts = [
    item.starred ? t('widgets.gallery.marks.bookmarked') : item.rejected ? t('widgets.gallery.marks.rejected') : null,
    item.rating > 0 ? t('widgets.gallery.marks.ratingSummary', { count: item.rating }) : null,
    item.colorLabel
      ? t('widgets.gallery.marks.colorLabel', { label: getGalleryColorLabelName(t, item.colorLabel) })
      : null,
  ].filter((part): part is string => part !== null);

  return parts.length > 0 ? parts.join(', ') : null;
};

/**
 * A label's dot on a plain surface: the hue itself, edged in the palette's text shade, which keeps 3:1 against light
 * surfaces where the pale hues alone do not. A ring in the same shade marks the chosen label.
 */
export const GalleryColorLabelDot = ({
  checked = false,
  colorLabel,
  ...boxProps
}: Omit<BoxProps, 'children'> & { checked?: boolean; colorLabel: GalleryColorLabel }) => {
  const palette = GALLERY_COLOR_LABEL_PALETTES[colorLabel];

  return (
    <Box
      {...boxProps}
      bg={`${palette}.swatch`}
      borderColor={`${palette}.fg`}
      borderWidth="1px"
      flexShrink={0}
      outline={checked ? '2px solid' : undefined}
      outlineColor={`${palette}.fg`}
      outlineOffset="2px"
      rounded="full"
    />
  );
};

type GalleryMarkGlyphProps = Omit<IconProps, 'as' | 'children' | 'color' | 'fill'>;

/** A rating star on a plain surface: gold, edged like a label dot; unfilled, it takes the text colour. */
export const GalleryRatingStar = ({ filled, ...iconProps }: GalleryMarkGlyphProps & { filled: boolean }) => (
  <Icon {...iconProps} as={StarIcon} color={filled ? 'gold.fg' : undefined} fill={filled ? 'gold.swatch' : 'none'} />
);

/** Filled in the text colour when set; an outline in the surrounding colour when not. */
export const GalleryBookmarkGlyph = ({ checked, ...iconProps }: GalleryMarkGlyphProps & { checked: boolean }) => (
  <Icon {...iconProps} as={BookmarkIcon} color={checked ? 'fg' : undefined} fill={checked ? 'currentColor' : 'none'} />
);

/** Coral, edged like the star, when set; an outline in the surrounding colour when not. */
export const GalleryRejectGlyph = ({ checked, ...iconProps }: GalleryMarkGlyphProps & { checked: boolean }) => (
  <Icon
    {...iconProps}
    as={FlagIcon}
    color={checked ? 'coral.fg' : undefined}
    fill={checked ? 'coral.swatch' : 'none'}
  />
);

const REJECTED_SCRIM_BG = 'rgb(0 0 0 / 55%)';
/** Separates the label bar from an image of a similar hue; over a dark image the hue itself stands out. */
const LABEL_BAR_KEYLINE = '0 -1px 0 rgb(0 0 0 / 60%)';

/**
 * Inside the media box: a scrim dims a rejected item and a bar along the bottom edge shows its label. Neither is the
 * only cue: the corner badge says rejected, and descriptions name the label.
 */
export const GalleryItemMarksMedia = ({ item }: { item: GalleryItemMarkFields }) => (
  <>
    {item.rejected ? (
      <Box aria-hidden="true" bg={REJECTED_SCRIM_BG} inset="0" pointerEvents="none" position="absolute" />
    ) : null}
    {item.colorLabel ? (
      <Box
        aria-hidden="true"
        bg={`${GALLERY_COLOR_LABEL_PALETTES[item.colorLabel]}.swatch`}
        bottom="0"
        boxShadow={LABEL_BAR_KEYLINE}
        h="3px"
        insetInline="0"
        pointerEvents="none"
        position="absolute"
      />
    ) : null}
  </>
);

/** Rating and rejected badges, non-interactive; nothing for an item with neither. */
export const GalleryItemMarksBadges = ({
  compact = false,
  item,
}: {
  /** Filmstrip thumbs: no badge chrome, glyphs only. */
  compact?: boolean;
  item: GalleryItemMarkFields;
}) => {
  if (!item.rejected && item.rating === 0) {
    return null;
  }

  return (
    <HStack
      aria-hidden="true"
      bottom={compact ? '0.5' : '1'}
      gap="0.5"
      insetInlineEnd={compact ? '0.5' : '1'}
      pointerEvents="none"
      position="absolute"
      zIndex="1"
    >
      {item.rejected ? (
        // The swatch, not the palette's solid, which light themes darken; a dark glyph stays legible on it.
        <Badge bg="coral.swatch" color="black" px={compact ? '0.5' : undefined} variant="solid">
          <Icon as={FlagIcon} boxSize={compact ? '2' : '2.5'} fill="currentColor" strokeWidth="2.5" />
        </Badge>
      ) : null}
      {item.rating > 0 ? (
        <Badge fontVariantNumeric="tabular-nums" gap="0.5" px={compact ? '0.5' : undefined} variant="solid">
          {/* Edged in the badge's text colour, which contrasts with its fill in every theme. */}
          <Icon as={StarIcon} boxSize={compact ? '2' : '2.5'} fill="gold.swatch" />
          {item.rating}
        </Badge>
      ) : null}
    </HStack>
  );
};

/**
 * Marks laid over a media button as its sibling, never inside it: text in the button would join its content and
 * contradict its accessible name. Nothing for an unmarked item.
 */
export const GalleryItemMarksOverlay = ({
  compact = false,
  item,
  ...boxProps
}: Omit<BoxProps, 'children'> & { compact?: boolean; item: GalleryItemMarkFields }) =>
  item.rejected || item.rating > 0 || item.colorLabel ? (
    <Box aria-hidden="true" overflow="hidden" pointerEvents="none" position="absolute" {...boxProps}>
      <GalleryItemMarksMedia item={item} />
      <GalleryItemMarksBadges compact={compact} item={item} />
    </Box>
  ) : null;
