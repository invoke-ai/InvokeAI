import type { CSSProperties, FocusEvent, KeyboardEvent, ReactNode, UIEvent } from 'react';

import { Box, Flex, Spinner } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui/Scrollable';
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useVirtualizer, type Range } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import type { ListDensity } from './ListItem';
import type { ListRow } from './listRows';

import { IN_SLOT_DIVIDER_HIDING_CSS, ListDivider } from './ListDivider';
import { type ListInset, LIST_ROW_GAP_PX as ROW_GAP_PX, LIST_ROW_INSET } from './listLayout';
import { LIST_SECTION_HEADER_HEIGHT_PX, ListSectionHeader } from './ListSectionHeader';
import { extractRangeKeepingPinned, getPinnedHeaderShift, lastHeaderAtOrBefore } from './virtualSections';

const ITEM_HEIGHT_PX: Record<ListDensity, number> = { comfortable: 52, compact: 28, regular: 40, snug: 48 };
const getItemPitch = (density: ListDensity, estimatedRowHeight?: number) =>
  (estimatedRowHeight ?? ITEM_HEIGHT_PX[density]) + ROW_GAP_PX;

export const getListContentHeight = (rows: readonly Pick<ListRow<unknown>, 'kind'>[], density: ListDensity): number =>
  rows.reduce(
    (total, row) => total + (row.kind === 'header' ? LIST_SECTION_HEADER_HEIGHT_PX : getItemPitch(density)),
    0
  );
const OVERSCAN_ROWS = 8;
const PRIMARY_SELECTOR = '[data-list-primary]';
// Set on the list root from `inset`; row and header wrappers pad by it and in-slot dividers offset from it.
const INSET = 'var(--list-row-inset)';

// Rows are fixed-height slots: content that outgrows its density clips instead of overlapping the next row.
const ITEM_WRAPPER_STYLE: CSSProperties = {
  display: 'grid',
  left: 0,
  overflow: 'hidden',
  paddingBottom: ROW_GAP_PX,
  paddingInline: INSET,
  position: 'absolute',
  top: 0,
  width: '100%',
};
const MEASURED_ITEM_WRAPPER_STYLE: CSSProperties = {
  left: 0,
  paddingBottom: ROW_GAP_PX,
  paddingInline: INSET,
  position: 'absolute',
  top: 0,
  width: '100%',
};
const HEADER_WRAPPER_STYLE: CSSProperties = {
  height: LIST_SECTION_HEADER_HEIGHT_PX,
  left: 0,
  paddingInline: INSET,
  position: 'absolute',
  top: 0,
  width: '100%',
};
const PINNED_HEADER_WRAPPER_STYLE: CSSProperties = {
  background: 'var(--list-surface)',
  height: LIST_SECTION_HEADER_HEIGHT_PX,
  paddingInline: INSET,
  position: 'sticky',
  top: 0,
  zIndex: 1,
};

/** Pinning the scroll area's inherited overflow properties here stops a per-scroll restyle of every mounted row. */
const ROW_CONTAINER_CSS = {
  '--scroll-area-overflow-x-start': '0px',
  '--scroll-area-overflow-x-end': '0px',
  '--scroll-area-overflow-y-start': '0px',
  '--scroll-area-overflow-y-end': '0px',
};
const ROW_CONTAINER_WITH_DIVIDERS_CSS = [ROW_CONTAINER_CSS, IN_SLOT_DIVIDER_HIDING_CSS];

export type ListSurface = 'bg' | 'bg.subtle' | 'bg.panel';

/** A row that is not a ListItem renders `role="listitem"` itself; the List does not track its focus. */
export interface ListRowProps {
  density: ListDensity;
  isActive: boolean;
  isBusy: boolean;
  itemKey: string;
  positionInSet: number;
  setSize: number;
  tabIndex: number;
}

export interface ListProps<T> {
  rows: readonly ListRow<T>[];
  label: string;
  renderItem: (item: T, rowProps: ListRowProps) => ReactNode;
  /** Key of the row whose detail is open; it is the initial keyboard target. */
  activeKey?: string | null;
  /** Keyboard target while no row was focused and none is active; defaults to the first row. */
  defaultFocusKey?: string | null;
  /** Row pitch; fixed for the list's lifetime, so change it with a keyed remount. */
  density?: ListDensity;
  /** `measured` sizes rows by content (density is then the estimate); fixed rows clip to the density's pitch. */
  rowHeight?: 'fixed' | 'measured';
  estimatedRowHeight?: number;
  status?: 'loading' | 'error' | 'ready';
  errorState?: ReactNode;
  /** Shown instead of rows when there are none; the caller decides between empty and no-match copy. */
  emptyState?: ReactNode;
  /** Rows belong to a request being replaced: announced with aria-busy and rendered inert. */
  isBusy?: boolean;
  /** The background pinned headers paint over rows with; match the surface the list sits on. */
  surface?: ListSurface;
  dividers?: boolean;
  inset?: ListInset;
  initialScrollOffset?: number;
  /** Needs its rows on first render, so it suits lists whose data is already in hand. */
  revealActiveOnMount?: boolean;
  /** Receives the final scroll offset on unmount; must be referentially stable. */
  onScrollOffsetPersist?: (offset: number) => void;
  /** Visible row-index range (headers and items share it), never per pixel; must be referentially stable. */
  onRangeChange?: (range: ListRange) => void;
}

export interface ListRange {
  startIndex: number;
  endIndex: number;
}

const findItemIndex = <T,>(rows: readonly ListRow<T>[], from: number, step: 1 | -1): number => {
  for (let index = from; index >= 0 && index < rows.length; index += step) {
    if (rows[index]?.kind === 'item') {
      return index;
    }
  }

  return -1;
};

const primaryOf = (viewport: HTMLElement, key: string): HTMLElement | null =>
  viewport.querySelector<HTMLElement>(`[data-list-row="${CSS.escape(key)}"] ${PRIMARY_SELECTOR}`);

/** Inputs whose arrows, Home and End do nothing of their own: a row's selection checkbox and button-like inputs. */
const KEYLESS_INPUT_TYPES = new Set(['button', 'checkbox', 'color', 'file', 'image', 'reset', 'submit']);

/** The List leaves these keys to a row's field, select, radio or range input. */
const usesNavigationKeys = (target: HTMLElement): boolean => {
  if (target.isContentEditable || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) {
    return true;
  }

  return target instanceof HTMLInputElement && !KEYLESS_INPUT_TYPES.has(target.type);
};

export const List = <T,>({
  activeKey = null,
  defaultFocusKey = null,
  density = 'regular',
  dividers = false,
  emptyState,
  errorState,
  estimatedRowHeight,
  initialScrollOffset,
  inset = 'row',
  isBusy = false,
  label,
  renderItem,
  revealActiveOnMount = false,
  rowHeight = 'fixed',
  rows,
  status = 'ready',
  surface = 'bg',
  onRangeChange,
  onScrollOffsetPersist,
}: ListProps<T>) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pendingFocus = useRef<string | null>(null);
  // Focus inside the list at the last focus event; a focused row that unmounts fires no blur, so this stays true.
  const listOwnsFocus = useRef(false);
  const lastScrollOffset = useRef(initialScrollOffset ?? 0);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const itemPitch = getItemPitch(density, rowHeight === 'measured' ? estimatedRowHeight : undefined);

  const layout = useMemo(() => {
    const headerIndexes: number[] = [];
    const indexByKey = new Map<string, number>();
    const ordinals: number[] = [];
    let firstItemKey: string | null = null;
    let itemCount = 0;

    rows.forEach((row, index) => {
      indexByKey.set(row.key, index);

      if (row.kind === 'header') {
        headerIndexes.push(index);
        ordinals.push(0);
      } else {
        firstItemKey ??= row.key;
        itemCount += 1;
        ordinals.push(itemCount);
      }
    });

    return { firstItemKey, headerIndexes, indexByKey, itemCount, ordinals };
  }, [rows]);
  const { firstItemKey, headerIndexes, indexByKey, itemCount, ordinals } = layout;
  const isMeasured = rowHeight === 'measured';
  const hasHeaders = headerIndexes.length > 0;

  // The roving target survives filtering: fall back to the active row, then the default one, then the first item.
  const focusKeyIsStale = focusKey !== null && !indexByKey.has(focusKey);
  const resolvedFocusKey =
    !focusKeyIsStale && focusKey !== null
      ? focusKey
      : activeKey !== null && indexByKey.has(activeKey)
        ? activeKey
        : defaultFocusKey !== null && indexByKey.has(defaultFocusKey)
          ? defaultFocusKey
          : firstItemKey;
  const focusIndex = resolvedFocusKey === null ? -1 : (indexByKey.get(resolvedFocusKey) ?? -1);

  // Keep the pinned header and the focused row mounted: focus must never fall off the list.
  const rangeExtractor = useCallback(
    (range: Range) => extractRangeKeepingPinned(range, headerIndexes, focusIndex),
    [focusIndex, headerIndexes]
  );
  const estimateSize = useCallback(
    (index: number) => (rows[index]?.kind === 'header' ? LIST_SECTION_HEADER_HEIGHT_PX : itemPitch),
    [itemPitch, rows]
  );
  const getItemKey = useCallback((index: number) => rows[index]?.key ?? index, [rows]);
  const activeIndex = revealActiveOnMount && activeKey !== null ? (indexByKey.get(activeKey) ?? -1) : -1;
  const getActiveRowOffset = useCallback(() => {
    let offset = 0;

    for (let index = 0; index < activeIndex; index += 1) {
      offset += estimateSize(index);
    }

    return Math.max(0, offset - (hasHeaders ? LIST_SECTION_HEADER_HEIGHT_PX : 0));
  }, [activeIndex, estimateSize, hasHeaders]);
  const getScrollElement = useCallback(() => viewportRef.current, []);
  const reportedRange = useRef<ListRange | null>(null);
  // The virtualizer also notifies for scroll start and stop and for measurements; report only a moved range.
  const handleVirtualizerChange = useCallback(
    (instance: { range: { startIndex: number; endIndex: number } | null }) => {
      const range = instance.range;
      const last = reportedRange.current;

      if (!onRangeChange || !range || (last?.startIndex === range.startIndex && last.endIndex === range.endIndex)) {
        return;
      }

      reportedRange.current = { endIndex: range.endIndex, startIndex: range.startIndex };
      onRangeChange(reportedRange.current);
    },
    [onRangeChange]
  );
  // An unmounting viewport takes no blur; forget ownership so a remount cannot pull focus back.
  const attachViewport = useCallback((element: HTMLDivElement | null) => {
    viewportRef.current = element;

    if (element === null) {
      listOwnsFocus.current = false;
    }
  }, []);
  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize,
    getItemKey,
    getScrollElement,
    initialOffset: initialScrollOffset ?? getActiveRowOffset,
    onChange: onRangeChange ? handleVirtualizerChange : undefined,
    overscan: OVERSCAN_ROWS,
    rangeExtractor,
    scrollPaddingStart: hasHeaders ? LIST_SECTION_HEADER_HEIGHT_PX : 0,
    // useSyncExternalStore already commits scroll snapshots synchronously; flushSync would only add React errors.
    useFlushSync: false,
  });
  const { scrollToIndex, virtualItems } = virtualizer;
  const scrollOffset = virtualizer.scrollOffset ?? 0;

  const pinnedIndex = hasHeaders ? lastHeaderAtOrBefore(headerIndexes, virtualizer.range?.startIndex ?? 0) : null;
  const nextHeaderIndex = pinnedIndex === null ? undefined : headerIndexes[headerIndexes.indexOf(pinnedIndex) + 1];
  const nextHeaderStart = virtualItems.find((virtualRow) => virtualRow.index === nextHeaderIndex)?.start;
  const pinnedStyle = useMemo(() => {
    const shift = getPinnedHeaderShift(nextHeaderStart, scrollOffset, LIST_SECTION_HEADER_HEIGHT_PX);

    return shift === 0
      ? PINNED_HEADER_WRAPPER_STYLE
      : { ...PINNED_HEADER_WRAPPER_STYLE, transform: `translateY(${shift}px)` };
  }, [nextHeaderStart, scrollOffset]);

  // Before paint so the ring never lags the scroll; also repairs focus a vanished row left on the body.
  useLayoutEffect(() => {
    const viewport = viewportRef.current;

    if (!viewport) {
      return;
    }

    const pending = pendingFocus.current;

    if (pending !== null) {
      if (!indexByKey.has(pending)) {
        pendingFocus.current = null;
      } else {
        const target = primaryOf(viewport, pending);

        if (target) {
          target.focus();
          pendingFocus.current = null;
        }

        return;
      }
    }

    if (
      focusKeyIsStale &&
      resolvedFocusKey !== null &&
      listOwnsFocus.current &&
      !viewport.contains(document.activeElement)
    ) {
      primaryOf(viewport, resolvedFocusKey)?.focus();
    }
  }, [focusKeyIsStale, indexByKey, resolvedFocusKey, virtualItems]);

  const moveFocus = useCallback(
    (index: number) => {
      const row = rows[index];

      if (!row) {
        return;
      }

      setFocusKey(row.key);
      pendingFocus.current = row.key;
      scrollToIndex(index, { align: 'auto' });
    },
    [rows, scrollToIndex]
  );
  const handleKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      const target = event.target as HTMLElement;

      if (event.defaultPrevented || !target.closest('[data-list-row]') || usesNavigationKeys(target)) {
        return;
      }

      let next: number;

      switch (event.key) {
        case 'ArrowDown':
          next = findItemIndex(rows, focusIndex + 1, 1);
          break;
        case 'ArrowUp':
          next = findItemIndex(rows, focusIndex - 1, -1);
          break;
        case 'Home':
          next = findItemIndex(rows, 0, 1);
          break;
        case 'End':
          next = findItemIndex(rows, rows.length - 1, -1);
          break;
        default:
          return;
      }

      event.preventDefault();

      if (next >= 0 && next !== focusIndex) {
        moveFocus(next);
      }
    },
    [focusIndex, moveFocus, rows]
  );
  // Pointer focus updates the roving target so Tab leaves from, and returns to, the row the user last used.
  const handleFocusCapture = useCallback((event: FocusEvent<HTMLDivElement>) => {
    listOwnsFocus.current = true;
    const key = (event.target as HTMLElement).closest('[data-list-row]')?.getAttribute('data-list-row');

    if (key) {
      setFocusKey(key);
    }
  }, []);
  const handleBlurCapture = useCallback((event: FocusEvent<HTMLDivElement>) => {
    listOwnsFocus.current = event.relatedTarget !== null && event.currentTarget.contains(event.relatedTarget);
  }, []);
  const handleScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    lastScrollOffset.current = event.currentTarget.scrollTop;
  }, []);
  const viewportProps = useMemo(
    () => ({
      'data-list-viewport': '',
      onBlurCapture: handleBlurCapture,
      onFocusCapture: handleFocusCapture,
      onKeyDown: handleKeyDown,
      onScroll: handleScroll,
      style: hasHeaders ? { scrollPaddingTop: LIST_SECTION_HEADER_HEIGHT_PX } : undefined,
    }),
    [handleBlurCapture, handleFocusCapture, handleKeyDown, handleScroll, hasHeaders]
  );

  useMountEffect(() => () => onScrollOffsetPersist?.(lastScrollOffset.current));

  const rootCss = useMemo(
    () => ({ '--list-row-inset': inset === 'none' ? '0px' : LIST_ROW_INSET, '--list-surface': `colors.${surface}` }),
    [inset, surface]
  );
  const containerStyle = useMemo<CSSProperties>(
    () => ({ height: virtualizer.totalSize, position: 'relative', width: '100%' }),
    [virtualizer.totalSize]
  );

  let content: ReactNode;

  if (status === 'loading') {
    content = (
      <Flex align="center" aria-label={t('common.loading')} h="full" justify="center" py="8" role="status" w="full">
        <Spinner color="fg.subtle" size="lg" />
      </Flex>
    );
  } else if (status === 'error') {
    content = (
      <Flex align="center" h="full" justify="center" minH="0" overflowY="auto" p="3" w="full">
        {errorState}
      </Flex>
    );
  } else if (itemCount === 0) {
    content = (
      <Flex align="center" h="full" justify="center" minH="0" overflowY="auto" p="3" w="full">
        {emptyState}
      </Flex>
    );
  } else {
    content = (
      <Scrollable h="full" viewportProps={viewportProps} viewportRef={attachViewport}>
        <Box
          aria-busy={isBusy || undefined}
          aria-label={label}
          css={dividers ? ROW_CONTAINER_WITH_DIVIDERS_CSS : ROW_CONTAINER_CSS}
          role="list"
          style={containerStyle}
        >
          {virtualItems.map((virtualRow) => {
            const row = rows[virtualRow.index];

            if (!row) {
              return null;
            }

            if (row.kind === 'header') {
              const isPinned = virtualRow.index === pinnedIndex;

              return (
                <div
                  key={virtualRow.key}
                  data-list-pinned-header={isPinned ? '' : undefined}
                  role="presentation"
                  style={
                    isPinned ? pinnedStyle : { ...HEADER_WRAPPER_STYLE, transform: `translateY(${virtualRow.start}px)` }
                  }
                >
                  <ListSectionHeader count={row.count} label={row.label} />
                </div>
              );
            }

            return (
              <div
                key={virtualRow.key}
                ref={isMeasured ? virtualizer.measureElement : undefined}
                data-index={virtualRow.index}
                role="presentation"
                style={
                  isMeasured
                    ? { ...MEASURED_ITEM_WRAPPER_STYLE, transform: `translateY(${virtualRow.start}px)` }
                    : { ...ITEM_WRAPPER_STYLE, height: itemPitch, transform: `translateY(${virtualRow.start}px)` }
                }
              >
                {renderItem(row.item, {
                  density,
                  isActive: row.key === activeKey,
                  isBusy,
                  itemKey: row.key,
                  positionInSet: ordinals[virtualRow.index]!,
                  setSize: itemCount,
                  tabIndex: row.key === resolvedFocusKey ? 0 : -1,
                })}
                {dividers && rows[virtualRow.index + 1]?.kind === 'item' ? <ListDivider placement="in-slot" /> : null}
              </div>
            );
          })}
        </Box>
      </Scrollable>
    );
  }

  return (
    <Box css={rootCss} flex="1" h="full" minH="0" position="relative" w="full">
      {content}
    </Box>
  );
};
