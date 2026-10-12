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
import { LIST_ROW_GAP_PX as ROW_GAP_PX, LIST_ROW_INSET as INSET } from './listLayout';
import { LIST_SECTION_HEADER_HEIGHT_PX, ListSectionHeader } from './ListSectionHeader';
import { extractRangeKeepingPinned, getPinnedHeaderShift, lastHeaderAtOrBefore } from './virtualSections';

const ITEM_HEIGHT_PX: Record<ListDensity, number> = { comfortable: 52, compact: 28, regular: 40, snug: 48 };
const OVERSCAN_ROWS = 8;
const PRIMARY_SELECTOR = '[data-list-primary]';

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

export type ListSurface = 'bg' | 'bg.subtle' | 'bg.panel';

/**
 * Spread onto the ListItem a renderItem returns; the List owns density, focus order and set semantics. A row that
 * is not a ListItem (an inline editor, say) renders `role="listitem"` with the set position itself and keeps its
 * own tab order; the List does not track its focus, so scrolling can still unmount it while it is being edited.
 */
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
  /** Accessible name of the list. */
  label: string;
  renderItem: (item: T, rowProps: ListRowProps) => ReactNode;
  /** Key of the row whose detail is open; it is the initial keyboard target. */
  activeKey?: string | null;
  /** Row pitch; fixed for the list's lifetime, so change it with a keyed remount. */
  density?: ListDensity;
  /**
   * `measured` sizes each row by its content, for rows that grow (inline editors, wrapped errors); density is then
   * only the estimate. Fixed rows clip to the density's pitch and never measure.
   */
  rowHeight?: 'fixed' | 'measured';
  /** Typical measured row height; a close estimate keeps the scrollbar steady before rows are measured. */
  estimatedRowHeight?: number;
  status?: 'loading' | 'error' | 'ready';
  /** Shown instead of rows while status is error. */
  errorState?: ReactNode;
  /** Shown instead of rows when there are none; the caller decides between empty and no-match copy. */
  emptyState?: ReactNode;
  /** Rows belong to a request being replaced: announced with aria-busy and rendered inert. */
  isBusy?: boolean;
  /** The background pinned headers paint over rows with; match the surface the list sits on. */
  surface?: ListSurface;
  /** Hairlines between rows of a section, inset from the edges. */
  dividers?: boolean;
  initialScrollOffset?: number;
  /**
   * Without an initial offset, open with the active row at the top: for lists reopened after a selection was made
   * elsewhere. Needs its rows on first render, so it suits lists whose data is already in hand.
   */
  revealActiveOnMount?: boolean;
  /** Receives the final scroll offset on unmount; must be referentially stable. */
  onScrollOffsetPersist?: (offset: number) => void;
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

/**
 * Virtualized list with pinned section headers and a roving tabindex. Headers and items share one flat index
 * space; the header at the top of the visible range renders in flow as a sticky element, pushed out by the next.
 */
export const List = <T,>({
  activeKey = null,
  density = 'regular',
  dividers = false,
  emptyState,
  errorState,
  estimatedRowHeight,
  initialScrollOffset,
  isBusy = false,
  label,
  renderItem,
  revealActiveOnMount = false,
  rowHeight = 'fixed',
  rows,
  status = 'ready',
  surface = 'bg',
  onScrollOffsetPersist,
}: ListProps<T>) => {
  const { t } = useTranslation();
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const pendingFocus = useRef<string | null>(null);
  // Focus inside the list at the last focus event; a focused row that unmounts fires no blur, so this stays true.
  const listOwnsFocus = useRef(false);
  const lastScrollOffset = useRef(initialScrollOffset ?? 0);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const itemPitch =
    (rowHeight === 'measured' && estimatedRowHeight !== undefined ? estimatedRowHeight : ITEM_HEIGHT_PX[density]) +
    ROW_GAP_PX;

  const layout = useMemo(() => {
    const headerIndexes: number[] = [];
    const indexByKey = new Map<string, number>();
    /** 1-based position among items for aria-posinset; 0 for headers. */
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

  // The roving target survives filtering: fall back to the active row, then the first item.
  const focusKeyIsStale = focusKey !== null && !indexByKey.has(focusKey);
  const resolvedFocusKey =
    !focusKeyIsStale && focusKey !== null
      ? focusKey
      : activeKey !== null && indexByKey.has(activeKey)
        ? activeKey
        : firstItemKey;
  const focusIndex = resolvedFocusKey === null ? -1 : (indexByKey.get(resolvedFocusKey) ?? -1);

  // Keep the pinned header and the focused row mounted whatever the scroll position: focus must never fall off
  // the list, and the sticky header is the one that scrolled past the top of the visible range.
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
  // A viewport that unmounts (empty or loading state) takes no blur with it; forget ownership so a later remount
  // cannot pull focus away from wherever the user went meanwhile.
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
    overscan: OVERSCAN_ROWS,
    rangeExtractor,
    scrollPaddingStart: hasHeaders ? LIST_SECTION_HEADER_HEIGHT_PX : 0,
    // Scrollable's phantom-scrollbar heal dispatches a scroll event from a mount effect, where flushSync is a
    // React error; useSyncExternalStore already commits scroll snapshots synchronously.
    useFlushSync: false,
  });
  const { scrollToIndex, virtualItems } = virtualizer;
  const scrollOffset = virtualizer.scrollOffset ?? 0;

  const pinnedIndex = hasHeaders ? lastHeaderAtOrBefore(headerIndexes, virtualizer.range?.startIndex ?? 0) : null;
  // The next header can only push the pinned one while it is near the top, which keeps it mounted; its live start
  // holds for measured rows as well as fixed ones.
  const nextHeaderIndex = pinnedIndex === null ? undefined : headerIndexes[headerIndexes.indexOf(pinnedIndex) + 1];
  const nextHeaderStart = virtualItems.find((virtualRow) => virtualRow.index === nextHeaderIndex)?.start;
  const pinnedStyle = useMemo(() => {
    const shift = getPinnedHeaderShift(nextHeaderStart, scrollOffset, LIST_SECTION_HEADER_HEIGHT_PX);

    return shift === 0
      ? PINNED_HEADER_WRAPPER_STYLE
      : { ...PINNED_HEADER_WRAPPER_STYLE, transform: `translateY(${shift}px)` };
  }, [nextHeaderStart, scrollOffset]);

  // Focus lands before paint so the ring never lags the scroll that revealed the row. The same pass repairs focus
  // a vanished row left on the body: keyboard users keep their place when a filter or a delete removes it.
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
      if (event.defaultPrevented || !(event.target as HTMLElement).closest('[data-list-row]')) {
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

  const rootCss = useMemo(() => ({ '--list-surface': `colors.${surface}` }), [surface]);
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
          css={dividers ? IN_SLOT_DIVIDER_HIDING_CSS : undefined}
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
