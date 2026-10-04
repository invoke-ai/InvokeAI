import type { SystemStyleObject } from '@chakra-ui/react';
import type { GalleryItem, GalleryItemKey } from '@features/gallery/core/items';
import type { CSSProperties, MouseEvent } from 'react';

import { Box, Icon, Skeleton } from '@chakra-ui/react';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { getGalleryColumnCountForCell } from '@features/gallery/ui/galleryGridLayout';
import { GalleryTileFrame } from '@features/gallery/ui/GalleryTileFrame';
import { Scrollable } from '@platform/ui/Scrollable';
import { CheckIcon } from 'lucide-react';
import { memo, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import {
  GALLERY_PICKER_CELL_PX,
  GALLERY_PICKER_MAX_COLUMNS,
  GALLERY_PICKER_MIN_COLUMNS,
  type GalleryPickerTileState,
} from './galleryPicker';

const GRID_GAP_PX = 4;
const SKELETON_TILE_COUNT = 8;

const IMG_STYLE: CSSProperties = {
  display: 'block',
  height: '100%',
  inset: 0,
  maxWidth: 'none',
  objectFit: 'cover',
  position: 'absolute',
  width: '100%',
};

const ACTIVE_TILE_CSS: SystemStyleObject = {
  outline: '2px solid {colors.accent.solid}',
  outlineOffset: '-4px',
};

// Tiles keep the default arrow like every other control; only full or unsupported ones say they won't pick.
const TILE_CSS: SystemStyleObject = { cursor: 'default' };
const INERT_TILE_CSS: SystemStyleObject = { cursor: 'not-allowed', opacity: 0.35 };

const getTileCss = (state: GalleryPickerTileState, isActive: boolean): SystemStyleObject => {
  const base = state === 'pickable' || state === 'added' ? TILE_CSS : INERT_TILE_CSS;

  return isActive ? { ...base, ...ACTIVE_TILE_CSS } : base;
};

export const galleryPickerOptionId = (idBase: string, key: GalleryItemKey): string => `${idBase}-${key}`;

const GalleryPickerTile = memo(function GalleryPickerTile({
  idBase,
  isActive,
  isCurrent,
  isMultiple,
  item,
  state,
}: {
  idBase: string;
  isActive: boolean;
  /** The Gallery widget's own selection, ringed so it reads as the default. */
  isCurrent: boolean;
  isMultiple: boolean;
  item: GalleryItem;
  state: GalleryPickerTileState;
}) {
  const { t } = useTranslation();
  const key = toGalleryItemKey(item);
  const css = useMemo(() => getTileCss(state, isActive), [isActive, state]);
  const unsupportedLabel =
    state === 'unsupported'
      ? t(item.kind === 'video' ? 'widgets.gallery.picker.unsupportedVideo' : 'widgets.gallery.picker.unsupportedImage')
      : undefined;

  // Active-state ref callbacks reveal keyboard highlights; pointer hover must not scroll.
  const scrollIntoView = useCallback(
    (node: HTMLDivElement | null) => {
      if (node && isActive) {
        node.scrollIntoView({ block: 'nearest' });
      }
    },
    [isActive]
  );

  return (
    <GalleryTileFrame
      ref={scrollIntoView}
      aria-disabled={state === 'pickable' ? undefined : true}
      aria-label={state === 'added' ? t('widgets.gallery.picker.addedItem', { name: item.name }) : item.name}
      // Single mode: selection follows the highlight (what Enter picks).
      // Multiple mode: selection is what has been added; the highlight is
      // carried by `aria-activedescendant` alone.
      aria-selected={isMultiple ? state === 'added' : isActive}
      css={css}
      data-item-key={key}
      id={galleryPickerOptionId(idBase, key)}
      isSelected={isCurrent || state === 'added'}
      item={item}
      role="option"
      title={unsupportedLabel}
    >
      <img
        alt=""
        decoding="async"
        draggable={false}
        loading="lazy"
        src={item.thumbnailUrl || item.fullUrl}
        style={IMG_STYLE}
      />
      {state === 'added' ? (
        <Box
          alignItems="center"
          bg="accent.solid"
          boxSize="4"
          color="accent.contrast"
          display="flex"
          insetInlineEnd="1"
          justifyContent="center"
          pointerEvents="none"
          position="absolute"
          rounded="full"
          top="1"
          zIndex="1"
        >
          <Icon as={CheckIcon} boxSize="2.5" strokeWidth="3" />
        </Box>
      ) : null}
    </GalleryTileFrame>
  );
});

/**
 * The bounded infinite window limits nonvirtualized rows; lazy images defer requests and the sentinel loads
 * further pages.
 */
export const GalleryPickerGrid = ({
  activeKey,
  columnCount,
  currentKey,
  getTileState,
  idBase,
  isMultiple,
  isStale,
  items,
  label,
  onActivate,
  onColumnCountChange,
  onLoadMore,
}: {
  activeKey: GalleryItemKey | null;
  columnCount: number;
  currentKey: GalleryItemKey | null;
  getTileState: (item: GalleryItem) => GalleryPickerTileState;
  idBase: string;
  isMultiple: boolean;
  /** `items` belong to the previous scope while the current one loads. */
  isStale: boolean;
  /** Null while nothing has loaded yet. */
  items: GalleryItem[] | null;
  label: string;
  onActivate: (item: GalleryItem) => void;
  onColumnCountChange: (columnCount: number) => void;
  onLoadMore: () => void;
}) => {
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const loadMoreObserverRef = useRef<IntersectionObserver | null>(null);
  const itemsByKey = useMemo(() => new Map(items?.map((item) => [toGalleryItemKey(item), item])), [items]);

  const measureRef = useCallback(
    (node: HTMLDivElement | null) => {
      resizeObserverRef.current?.disconnect();
      resizeObserverRef.current = null;

      if (!node) {
        return;
      }

      const observer = new ResizeObserver(([entry]) => {
        const widthPx = entry?.contentRect.width ?? 0;

        if (widthPx > 0) {
          onColumnCountChange(
            getGalleryColumnCountForCell({
              max: GALLERY_PICKER_MAX_COLUMNS,
              min: GALLERY_PICKER_MIN_COLUMNS,
              targetCellPx: GALLERY_PICKER_CELL_PX,
              widthPx,
            })
          );
        }
      });

      observer.observe(node);
      resizeObserverRef.current = observer;
    },
    [onColumnCountChange]
  );

  // The scroll viewport is found from the DOM rather than a ref: the sentinel
  // can mount in the same commit as the viewport, before any parent ref is set.
  const sentinelRef = useCallback(
    (node: HTMLDivElement | null) => {
      loadMoreObserverRef.current?.disconnect();
      loadMoreObserverRef.current = null;

      if (!node) {
        return;
      }

      const observer = new IntersectionObserver(
        ([entry]) => {
          if (entry?.isIntersecting) {
            onLoadMore();
          }
        },
        { root: node.closest('[data-scope="scroll-area"][data-part="viewport"]'), rootMargin: '160px' }
      );

      observer.observe(node);
      loadMoreObserverRef.current = observer;
    },
    [onLoadMore]
  );

  const handleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      const key = (event.target as HTMLElement).closest<HTMLElement>('[data-item-key]')?.dataset.itemKey;
      const item = key ? itemsByKey.get(key as GalleryItemKey) : undefined;

      if (item) {
        onActivate(item);
      }
    },
    [itemsByKey, onActivate]
  );

  return (
    <Scrollable flex="1" minH="0">
      <Box
        ref={measureRef}
        aria-busy={items === null || isStale || undefined}
        aria-label={label}
        aria-multiselectable={isMultiple || undefined}
        display="grid"
        gap={`${GRID_GAP_PX}px`}
        gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
        id={idBase}
        opacity={isStale ? 0.6 : undefined}
        p="2"
        role="listbox"
        transition="opacity var(--wb-motion-duration-fast) ease"
        onClick={handleClick}
      >
        {items === null
          ? Array.from({ length: SKELETON_TILE_COUNT }, (_, index) => (
              <Skeleton key={index} aspectRatio={1} rounded="md" />
            ))
          : items.map((item) => {
              const key = toGalleryItemKey(item);

              return (
                <GalleryPickerTile
                  key={key}
                  idBase={idBase}
                  isActive={key === activeKey}
                  isCurrent={key === currentKey}
                  isMultiple={isMultiple}
                  item={item}
                  state={getTileState(item)}
                />
              );
            })}
      </Box>
      {items && items.length > 0 && !isStale ? <Box ref={sentinelRef} aria-hidden="true" h="1px" /> : null}
    </Scrollable>
  );
};
