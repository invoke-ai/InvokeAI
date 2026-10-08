import type { SystemStyleObject } from '@chakra-ui/react';
import type { GalleryItem, GalleryItemKey } from '@features/gallery/core/items';
import type { GallerySparsePageState } from '@features/gallery/ui/useGalleryData';
import type { CSSProperties, MouseEvent } from 'react';

import { Box, Icon, Skeleton, Stack, Text } from '@chakra-ui/react';
import { toGalleryItemKey } from '@features/gallery/core/items';
import { GALLERY_PAGE_SIZE } from '@features/gallery/data/queries';
import { getGalleryColumnCountForCell } from '@features/gallery/ui/galleryGridLayout';
import { GalleryTileFrame } from '@features/gallery/ui/GalleryTileFrame';
import { Button } from '@platform/ui/Button';
import { Scrollable } from '@platform/ui/Scrollable';
import { CheckIcon } from 'lucide-react';
import { memo, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import {
  GALLERY_PICKER_CELL_PX,
  GALLERY_PICKER_MAX_COLUMNS,
  GALLERY_PICKER_MIN_COLUMNS,
  type GalleryPickerTileState,
} from './galleryPicker';

const GRID_GAP_PX = 4;
const GRID_PADDING_PX = 8;
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

export const galleryPickerOptionId = (idBase: string, index: number): string => `${idBase}-slot-${index}`;

const GalleryPickerTile = memo(function GalleryPickerTile({
  activeIndex,
  currentKey,
  idBase,
  index,
  isMultiple,
  item,
  state,
  total,
}: {
  activeIndex: number;
  currentKey: GalleryItemKey | null;
  idBase: string;
  index: number;
  isMultiple: boolean;
  item: GalleryItem;
  state: GalleryPickerTileState;
  total: number;
}) {
  const { t } = useTranslation();
  const key = toGalleryItemKey(item);
  const css = useMemo(() => getTileCss(state, index === activeIndex), [activeIndex, index, state]);
  const unsupportedLabel =
    state === 'unsupported'
      ? t(item.kind === 'video' ? 'widgets.gallery.picker.unsupportedVideo' : 'widgets.gallery.picker.unsupportedImage')
      : undefined;

  return (
    <GalleryTileFrame
      aria-disabled={state === 'pickable' ? undefined : true}
      aria-label={state === 'added' ? t('widgets.gallery.picker.addedItem', { name: item.name }) : item.name}
      aria-posinset={index + 1}
      aria-selected={isMultiple ? state === 'added' : index === activeIndex}
      aria-setsize={total}
      css={css}
      data-item-index={index}
      data-item-key={key}
      id={galleryPickerOptionId(idBase, index)}
      isSelected={key === currentKey || state === 'added'}
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

const GalleryPickerPlaceholder = ({
  error,
  idBase,
  index,
  isActive,
  isLoading,
  retry,
  total,
}: {
  error: Error | null;
  idBase: string;
  index: number;
  isActive: boolean;
  isLoading: boolean;
  retry?: () => Promise<unknown>;
  total: number;
}) => {
  const { t } = useTranslation();
  const handleRetry = useCallback(() => {
    if (retry) {
      void retry();
    }
  }, [retry]);

  return (
    <Box
      aria-busy={isLoading || undefined}
      aria-disabled="true"
      aria-label={error ? t('common.error') : isLoading ? t('common.loading') : undefined}
      aria-posinset={index + 1}
      aria-selected={false}
      aria-setsize={total}
      bg={isLoading ? 'bg.subtle' : undefined}
      borderColor="border.subtle"
      borderWidth="2px"
      css={isActive ? ACTIVE_TILE_CSS : undefined}
      data-item-index={index}
      display="flex"
      alignItems="center"
      justifyContent="center"
      id={galleryPickerOptionId(idBase, index)}
      minW="0"
      overflow="hidden"
      role="option"
      rounded="md"
    >
      {error && retry ? (
        <Stack align="center" gap="1" maxW="full" px="1">
          <Text color="fg.muted" fontSize="xs" lineClamp={2} textAlign="center">
            {error.message}
          </Text>
          <Button size="sm" variant="ghost" onClick={handleRetry}>
            {t('common.retry')}
          </Button>
        </Stack>
      ) : isLoading ? (
        <Skeleton aspectRatio={1} h="full" rounded="md" w="full" />
      ) : null}
    </Box>
  );
};

/** A virtualized absolute-slot picker backed by the same 60-item Query pages as Gallery. */
export const GalleryPickerGrid = ({
  activeIndex,
  columnCount,
  currentKey,
  getTileState,
  idBase,
  isMultiple,
  isStale,
  itemSlots,
  label,
  onActivate,
  onColumnCountChange,
  onVisibleRangeChange,
  pageStates,
  suppressInlineRetry,
  total,
}: {
  activeIndex: number;
  columnCount: number;
  currentKey: GalleryItemKey | null;
  getTileState: (item: GalleryItem) => GalleryPickerTileState;
  idBase: string;
  isMultiple: boolean;
  /** Slots belong to the previous scope while the current listing loads. */
  isStale: boolean;
  itemSlots: ReadonlyMap<number, GalleryItem>;
  label: string;
  onActivate: (item: GalleryItem) => void;
  onColumnCountChange: (columnCount: number) => void;
  onVisibleRangeChange: (range: { endIndexExclusive: number; startIndex: number }) => void;
  pageStates: ReadonlyMap<number, GallerySparsePageState>;
  suppressInlineRetry: boolean;
  total: number | null;
}) => {
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [viewportWidth, setViewportWidth] = useState(0);
  const totalSlots = total ?? SKELETON_TILE_COUNT;
  const rowCount = Math.ceil(totalSlots / columnCount);
  const rowPitch =
    viewportWidth > 0
      ? Math.max(1, (viewportWidth - GRID_PADDING_PX * 2 - GRID_GAP_PX * (columnCount - 1)) / columnCount) + GRID_GAP_PX
      : GALLERY_PICKER_CELL_PX + GRID_GAP_PX;
  const itemsByKey = useMemo(
    () => new Map([...itemSlots.values()].map((item) => [toGalleryItemKey(item), item])),
    [itemSlots]
  );

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
          setViewportWidth(widthPx);
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

  const getScrollElement = useCallback(() => viewportRef.current, []);
  const getItemKey = useCallback((index: number) => index, []);
  const estimateSize = useCallback(() => rowPitch, [rowPitch]);
  const handleVirtualizerChange = useCallback(
    (instance: { getVirtualItems: () => readonly { index: number }[] }) => {
      if (total === 0) {
        return;
      }

      const visibleRows = instance.getVirtualItems();
      const firstRow = visibleRows[0]?.index;
      const lastRow = visibleRows[visibleRows.length - 1]?.index;

      if (firstRow === undefined || lastRow === undefined) {
        return;
      }

      const startIndex = Math.min(total ?? GALLERY_PAGE_SIZE, firstRow * columnCount);
      const endIndexExclusive = Math.min(total ?? GALLERY_PAGE_SIZE, (lastRow + 1) * columnCount);

      onVisibleRangeChange({ endIndexExclusive, startIndex });
    },
    [columnCount, onVisibleRangeChange, total]
  );
  const virtualizer = useVirtualizer({
    count: rowCount,
    estimateSize,
    getItemKey,
    getScrollElement,
    onChange: handleVirtualizerChange,
    overscan: 2,
    useFlushSync: false,
  });
  const { measure, scrollToIndex, totalSize, virtualItems } = virtualizer;
  const isInitialPageFailed = pageStates.get(0)?.error !== null && pageStates.get(0)?.error !== undefined;
  const isAnyPageLoading = [...pageStates.values()].some((pageState) => pageState.isLoading);

  useLayoutEffect(() => {
    measure();
  }, [columnCount, measure, rowPitch]);

  useLayoutEffect(() => {
    if (activeIndex >= 0 && activeIndex < totalSlots) {
      scrollToIndex(Math.floor(activeIndex / columnCount), { align: 'auto' });
    }
  }, [activeIndex, columnCount, scrollToIndex, totalSlots]);

  const handleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      // Placeholder tiles from the previous scope are shown, not offered.
      if (isStale) {
        return;
      }

      const key = (event.target as HTMLElement).closest<HTMLElement>('[data-item-key]')?.dataset.itemKey;
      const item = key ? itemsByKey.get(key as GalleryItemKey) : undefined;

      if (item) {
        onActivate(item);
      }
    },
    [isStale, itemsByKey, onActivate]
  );

  return (
    <Scrollable flex="1" minH="0" viewportRef={viewportRef}>
      <Box
        ref={measureRef}
        aria-busy={isStale || (total === null && !isInitialPageFailed) || isAnyPageLoading || undefined}
        aria-label={label}
        aria-multiselectable={isMultiple || undefined}
        id={idBase}
        minW="0"
        opacity={isStale ? 0.6 : undefined}
        position="relative"
        role="listbox"
        transition="opacity var(--wb-motion-duration-fast) ease"
        w="full"
        onClick={handleClick}
        h={`${totalSize + GRID_PADDING_PX * 2}px`}
      >
        {virtualItems.map((virtualRow) => {
          const startIndex = virtualRow.index * columnCount;
          const endIndex = Math.min(totalSlots, startIndex + columnCount);
          const cells = Array.from({ length: endIndex - startIndex }, (_, offset) => startIndex + offset);

          return (
            <Box
              key={virtualRow.key}
              display="grid"
              gap={`${GRID_GAP_PX}px`}
              gridTemplateColumns={`repeat(${columnCount}, minmax(0, 1fr))`}
              h={`${rowPitch}px`}
              left="0"
              position="absolute"
              px={`${GRID_PADDING_PX}px`}
              pb={`${GRID_GAP_PX}px`}
              boxSizing="border-box"
              top="0"
              transform={`translateY(${virtualRow.start + GRID_PADDING_PX}px)`}
              w="full"
            >
              {cells.map((index) => {
                const item = itemSlots.get(index);

                if (item) {
                  return (
                    <GalleryPickerTile
                      key={index}
                      activeIndex={activeIndex}
                      currentKey={currentKey}
                      idBase={idBase}
                      index={index}
                      isMultiple={isMultiple}
                      item={item}
                      state={getTileState(item)}
                      total={total ?? totalSlots}
                    />
                  );
                }

                const pageOffset = Math.floor(index / GALLERY_PAGE_SIZE) * GALLERY_PAGE_SIZE;
                const pageState = pageStates.get(pageOffset);
                const hasError = pageState?.error !== null && pageState?.error !== undefined;
                const isRetrySlot = index === pageOffset && hasError;

                return (
                  <GalleryPickerPlaceholder
                    key={index}
                    error={isRetrySlot && !suppressInlineRetry ? pageState.error : null}
                    idBase={idBase}
                    index={index}
                    isActive={index === activeIndex}
                    isLoading={pageState ? pageState.isLoading : true}
                    retry={isRetrySlot && !suppressInlineRetry ? pageState.retry : undefined}
                    total={total ?? -1}
                  />
                );
              })}
            </Box>
          );
        })}
      </Box>
    </Scrollable>
  );
};
