import type { InputProps } from '@chakra-ui/react';
import type { ChangeEvent, CSSProperties, KeyboardEvent, MouseEvent, ReactNode, Ref } from 'react';

import { Box, createRecipeContext, HStack, Icon, InputGroup, ScrollArea, Spacer, Stack, Text } from '@chakra-ui/react';
import { isImeComposing } from '@platform/browser/imeComposition';
import { dropdownGroupLabel } from '@theme/recipes';
import { SearchIcon } from 'lucide-react';
import {
  Fragment,
  useCallback,
  useDeferredValue,
  useId,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useVirtualizer, type Range, type VirtualItem } from 'react-hook-tanstack-virtual';
import { useTranslation } from 'react-i18next';

import { Button } from './Button';
import { extractRangeKeepingPinned, getPinnedHeaderShift, lastHeaderAtOrBefore } from './list/virtualSections';

export interface PickerGroup<T> {
  id: string;
  name: string;
  options: T[];
  shortName?: string;
  colorPalette?: string;
  getCountLabel?: (count: number) => string;
}

export interface PickerOptionState {
  isActive: boolean;
  isCompact: boolean;
  isSelected: boolean;
}

/** Stands in for the options while there is nothing to choose from yet; announced, never listed as an option. */
export interface PickerStatus {
  message: string;
  isError?: boolean;
  /** Offers a retry under a failure. */
  onRetry?: () => void;
}

type PickerRow<T> =
  | { kind: 'header'; group: PickerGroup<T>; key: string }
  | {
      kind: 'option';
      group: PickerGroup<T>;
      id: string;
      isDisabled: boolean;
      key: string;
      option: T;
      positionInSet: number;
      setSize: number;
    };

/** Consecutive rendered rows: a header on its own, or the options of one group. */
type RowRun<T> =
  | { item: VirtualItem; kind: 'header' }
  | { group: PickerGroup<T>; items: VirtualItem[]; kind: 'options' };

interface PickerListboxHandle {
  scrollToIndex: (index: number) => void;
}

const SEARCH_ICON = <Icon as={SearchIcon} size="md" />;
const LIST_PADDING_PX = 4;
const HEADER_ESTIMATE_PX = 28;
const OPTION_ESTIMATE_PX = 36;
const COMPACT_OPTION_ESTIMATE_PX = 24;
const OVERSCAN_ROWS = 8;
// The list mounts as the popup opens, before its viewport is measured; assuming the 18rem cap renders a full page,
// active option included, in that first commit.
const INITIAL_RECT = { height: 288, width: 0 };
const ROW_STYLE: CSSProperties = { left: 0, position: 'absolute', top: 0, width: '100%' };
const PINNED_HEADER_STYLE: CSSProperties = { position: 'sticky', top: 0, zIndex: 1 };

// Chakra's Input adopts an enclosing Field's id, description and invalid state. The search box belongs to the popup,
// not to the field whose trigger opened it, so it renders the same recipe on a plain input.
const { withContext: withInputRecipe } = createRecipeContext({ key: 'input' });
const SearchInput = withInputRecipe<HTMLInputElement, InputProps>('input');

/** Option ids contain arbitrary characters; encoding keeps the DOM id a single IDREF token. */
const getOptionDomId = (listboxId: string, optionId: string): string =>
  `${listboxId}-option-${encodeURIComponent(optionId)}`;

const keepSearchFocus = (event: MouseEvent) => event.preventDefault();

const toRowRuns = <T,>(virtualItems: readonly VirtualItem[], rows: readonly PickerRow<T>[]): RowRun<T>[] => {
  const runs: RowRun<T>[] = [];

  for (const item of virtualItems) {
    const row = rows[item.index];

    if (!row) {
      continue;
    }

    if (row.kind === 'header') {
      runs.push({ item, kind: 'header' });
      continue;
    }

    const last = runs[runs.length - 1];

    if (last?.kind === 'options' && last.group === row.group) {
      last.items.push(item);
    } else {
      runs.push({ group: row.group, items: [item], kind: 'options' });
    }
  }

  return runs;
};

/**
 * Searchable single-choice list for a popup: an editable combobox that keeps focus while the arrows move a
 * virtualized listbox's active option. The active option always stays mounted, so `aria-activedescendant` never
 * names a missing element. Mount it with the popup: each mount starts from an empty search.
 */
export const Picker = <T,>({
  emptyMessage,
  getIsOptionDisabled,
  getOptionId,
  groups,
  isCompact = false,
  isMatch,
  listLabel,
  noMatchesMessage,
  renderOption,
  searchPlaceholder,
  searchSlot,
  selectedId,
  status,
  toolbarSlot,
  onSelect,
}: {
  emptyMessage: string;
  getIsOptionDisabled?: (option: T) => boolean;
  getOptionId: (option: T) => string;
  groups: PickerGroup<T>[];
  isCompact?: boolean;
  isMatch: (option: T, searchTerm: string) => boolean;
  listLabel: string;
  noMatchesMessage: string;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  searchPlaceholder: string;
  searchSlot?: ReactNode;
  selectedId: string | null;
  /** Replaces the list, e.g. while loading or after a failure. */
  status?: PickerStatus;
  toolbarSlot?: ReactNode;
  onSelect: (option: T) => void;
}) => {
  const { t } = useTranslation();
  const listboxId = useId();
  const [searchTerm, setSearchTerm] = useState('');
  const [activeId, setActiveId] = useState<string | null>(null);
  const deferredSearchTerm = useDeferredValue(searchTerm);
  const listboxRef = useRef<PickerListboxHandle>(null);
  const showHeaders = groups.length > 1;

  const visibleGroups = useMemo(() => {
    const term = deferredSearchTerm.trim();

    if (!term) {
      return groups.filter((group) => group.options.length > 0);
    }

    return groups
      .map((group) => ({ ...group, options: group.options.filter((option) => isMatch(option, term)) }))
      .filter((group) => group.options.length > 0);
  }, [deferredSearchTerm, groups, isMatch]);

  const { headerIndexes, indexById, rowKeys, rows, selectableIds } = useMemo(() => {
    const nextRows: PickerRow<T>[] = [];
    const nextHeaderIndexes: number[] = [];
    const nextIndexById = new Map<string, number>();
    const nextSelectableIds: string[] = [];

    for (const group of visibleGroups) {
      if (showHeaders) {
        nextHeaderIndexes.push(nextRows.length);
        nextRows.push({ group, key: `header:${group.id}`, kind: 'header' });
      }

      group.options.forEach((option, index) => {
        const id = getOptionId(option);
        const isDisabled = getIsOptionDisabled?.(option) ?? false;

        nextIndexById.set(id, nextRows.length);

        if (!isDisabled) {
          nextSelectableIds.push(id);
        }

        nextRows.push({
          group,
          id,
          isDisabled,
          key: `option:${id}`,
          kind: 'option',
          option,
          positionInSet: index + 1,
          setSize: group.options.length,
        });
      });
    }

    return {
      headerIndexes: nextHeaderIndexes,
      indexById: nextIndexById,
      // Row keys already encode group boundaries (headers) and option order; option data is left out on purpose.
      rowKeys: nextRows.map((row) => row.key).join('\u0000'),
      rows: nextRows,
      selectableIds: nextSelectableIds,
    };
  }, [getIsOptionDisabled, getOptionId, showHeaders, visibleGroups]);

  // A new result set remounts the listbox, which opens scrolled to its active option (see PickerListbox). Callers
  // rebuild groups on unrelated renders, so the set is compared by content: the search that produced it and its
  // ordered row keys. Equal content keeps the mounted list, its scroll position and rows, and data-only changes
  // re-render those rows in place. Built once per rows change; renders with unchanged rows compare one reference.
  const resultKey = useMemo(() => `${deferredSearchTerm}\u0001${rowKeys}`, [deferredSearchTerm, rowKeys]);
  const [listMount, setListMount] = useState({ generation: 0, resultKey });

  if (listMount.resultKey !== resultKey) {
    setListMount({ generation: listMount.generation + 1, resultKey });
  }

  const isSelectable = (id: string | null): id is string => {
    const row = id === null ? undefined : rows[indexById.get(id) ?? -1];

    return row?.kind === 'option' && !row.isDisabled;
  };
  // If filtering removes the active row, highlight the selection or else the first remaining result.
  const resolvedActiveId = isSelectable(activeId)
    ? activeId
    : isSelectable(selectedId)
      ? selectedId
      : (selectableIds[0] ?? null);
  const activeIndex = resolvedActiveId === null ? -1 : (indexById.get(resolvedActiveId) ?? -1);
  const hasAnyOption = groups.some((group) => group.options.length > 0);
  const isListShown = !status && rows.length > 0;
  const message = status?.message ?? (!hasAnyOption ? emptyMessage : isListShown ? null : noMatchesMessage);

  const moveActive = useCallback(
    (delta: 1 | -1) => {
      if (selectableIds.length === 0) {
        return;
      }

      const currentIndex = resolvedActiveId ? selectableIds.indexOf(resolvedActiveId) : -1;
      const nextIndex = Math.min(Math.max(currentIndex + delta, 0), selectableIds.length - 1);
      const nextId = selectableIds[nextIndex] ?? null;

      setActiveId(nextId);

      if (nextId) {
        listboxRef.current?.scrollToIndex(indexById.get(nextId) ?? 0);
      }
    },
    [indexById, resolvedActiveId, selectableIds]
  );

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      // Stop typing events before they reach the window command runtime.
      event.stopPropagation();

      if (isImeComposing(event.nativeEvent) || !isListShown) {
        return;
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault();
        moveActive(event.key === 'ArrowDown' ? 1 : -1);
        return;
      }

      const activeRow = rows[activeIndex];

      if (event.key === 'Enter' && activeRow?.kind === 'option') {
        event.preventDefault();
        onSelect(activeRow.option);
      }
    },
    [activeIndex, isListShown, moveActive, onSelect, rows]
  );

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(event.currentTarget.value);
    setActiveId(null);
  }, []);

  return (
    <Stack gap="0" minH="0">
      <Stack gap="2" p="2">
        <HStack gap="1">
          <InputGroup flex="1" minW="0" startElement={SEARCH_ICON}>
            <SearchInput
              // The listbox always keeps its active option mounted (see extractRangeKeepingPinned).
              aria-activedescendant={
                isListShown && resolvedActiveId ? getOptionDomId(listboxId, resolvedActiveId) : undefined
              }
              aria-autocomplete="list"
              aria-controls={isListShown ? listboxId : undefined}
              aria-expanded={isListShown}
              aria-label={searchPlaceholder}
              autoComplete="off"
              placeholder={searchPlaceholder}
              role="combobox"
              value={searchTerm}
              onChange={handleSearchChange}
              onKeyDown={handleSearchKeyDown}
            />
          </InputGroup>
          {searchSlot}
        </HStack>
        {toolbarSlot}
      </Stack>
      <Box borderColor="border.subtle" borderTopWidth="1px" />
      {isListShown ? (
        <PickerListbox<T>
          key={listMount.generation}
          ref={listboxRef}
          activeIndex={activeIndex}
          headerIndexes={headerIndexes}
          isCompact={isCompact}
          listboxId={listboxId}
          listLabel={listLabel}
          renderOption={renderOption}
          rows={rows}
          selectedId={selectedId}
          showHeaders={showHeaders}
          onActivate={setActiveId}
          onSelect={onSelect}
        />
      ) : null}
      {/* Mounted throughout so assistive technology hears loading, empty and failure messages change. */}
      <Box role="status">
        {message ? (
          <Text
            color={status?.isError ? 'fg.error' : 'fg.subtle'}
            fontSize="xs"
            pb={status?.onRetry ? '1.5' : '3'}
            pt="3"
            px="2"
          >
            {message}
          </Text>
        ) : null}
      </Box>
      {status?.onRetry ? (
        <Box pb="3" px="2">
          <Button size="sm" variant="outline" onClick={status.onRetry}>
            {t('common.retry')}
          </Button>
        </Box>
      ) : null}
    </Stack>
  );
};

/**
 * One result set's virtualized listbox. It is keyed by the result set, so each mount opens with the active option
 * in view: the virtualizer's initial offset places it from estimates, as Platform List's `revealActiveOnMount` does,
 * and a pre-paint reveal corrects for measured row heights. Keyboard moves within the set scroll through the handle.
 */
const PickerListbox = <T,>({
  activeIndex,
  headerIndexes,
  isCompact,
  listboxId,
  listLabel,
  ref,
  renderOption,
  rows,
  selectedId,
  showHeaders,
  onActivate,
  onSelect,
}: {
  activeIndex: number;
  headerIndexes: readonly number[];
  isCompact: boolean;
  listboxId: string;
  listLabel: string;
  ref: Ref<PickerListboxHandle>;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  rows: readonly PickerRow<T>[];
  selectedId: string | null;
  showHeaders: boolean;
  onActivate: (id: string) => void;
  onSelect: (option: T) => void;
}) => {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const headerHeight = showHeaders ? HEADER_ESTIMATE_PX : 0;
  const rangeExtractor = useCallback(
    (range: Range) => extractRangeKeepingPinned(range, headerIndexes, activeIndex),
    [activeIndex, headerIndexes]
  );
  const estimateSize = useCallback(
    (index: number) =>
      rows[index]?.kind === 'header' ? HEADER_ESTIMATE_PX : isCompact ? COMPACT_OPTION_ESTIMATE_PX : OPTION_ESTIMATE_PX,
    [isCompact, rows]
  );
  // Leave the list at the top when the active option already shows there; otherwise put it under the pinned header.
  const getInitialOffset = useCallback(() => {
    let start = LIST_PADDING_PX;

    for (let index = 0; index < activeIndex; index += 1) {
      start += estimateSize(index);
    }

    return start + estimateSize(activeIndex) <= INITIAL_RECT.height ? 0 : Math.max(0, start - headerHeight);
  }, [activeIndex, estimateSize, headerHeight]);
  const getItemKey = useCallback((index: number) => rows[index]?.key ?? index, [rows]);
  const getScrollElement = useCallback(() => viewportRef.current, []);
  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize,
    getItemKey,
    getScrollElement,
    initialOffset: getInitialOffset,
    initialRect: INITIAL_RECT,
    overscan: OVERSCAN_ROWS,
    paddingEnd: LIST_PADDING_PX,
    paddingStart: LIST_PADDING_PX,
    rangeExtractor,
    scrollPaddingStart: headerHeight,
    // A mount-time scroll event would make flushSync a React error; useSyncExternalStore already commits
    // scroll snapshots synchronously.
    useFlushSync: false,
  });
  const { measureElement, scrollToIndex, virtualItems } = virtualizer;
  // The option active when this result set mounted; later keyboard moves scroll through the handle.
  const [revealIndex] = useState(activeIndex);

  // The initial offset comes from estimates, and the rows above, at or below the active option may measure taller or
  // shorter, leaving it partly hidden. The first commit's rows are measured before layout effects run, so this nudges
  // it fully into view before paint; the virtualizer then keeps the alignment exact while newly shown rows measure.
  useLayoutEffect(() => {
    if (revealIndex >= 0) {
      scrollToIndex(revealIndex, { align: 'auto' });
    }
  }, [revealIndex, scrollToIndex]);

  useImperativeHandle(ref, () => ({ scrollToIndex: (index) => scrollToIndex(index, { align: 'auto' }) }), [
    scrollToIndex,
  ]);

  const listboxStyle = useMemo<CSSProperties>(
    () => ({ height: virtualizer.totalSize, paddingTop: LIST_PADDING_PX }),
    [virtualizer.totalSize]
  );
  const pinnedIndex = showHeaders ? lastHeaderAtOrBefore(headerIndexes, virtualizer.range?.startIndex ?? 0) : null;
  const nextHeaderIndex = pinnedIndex === null ? undefined : headerIndexes[headerIndexes.indexOf(pinnedIndex) + 1];
  const pinnedShift = getPinnedHeaderShift(
    virtualItems.find((item) => item.index === nextHeaderIndex)?.start,
    virtualizer.scrollOffset ?? 0,
    virtualItems.find((item) => item.index === pinnedIndex)?.size ?? HEADER_ESTIMATE_PX
  );

  const renderOptionRow = (item: VirtualItem) => {
    const row = rows[item.index];

    if (row?.kind !== 'option') {
      return null;
    }

    return (
      <PickerOptionRow
        key={item.key}
        domId={getOptionDomId(listboxId, row.id)}
        id={row.id}
        index={item.index}
        isActive={item.index === activeIndex}
        isCompact={isCompact}
        isDisabled={row.isDisabled}
        isSelected={selectedId === row.id}
        measureElement={measureElement}
        option={row.option}
        positionInSet={row.positionInSet}
        railColor={showHeaders ? getRailColor(row.group) : undefined}
        renderOption={renderOption}
        setSize={row.setSize}
        start={item.start}
        onActivate={onActivate}
        onSelect={onSelect}
      />
    );
  };

  return (
    <ScrollArea.Root maxH="18rem" variant="hover" w="full">
      <ScrollArea.Viewport ref={viewportRef} maxH="inherit" w="full">
        <ScrollArea.Content maxW="full" minW="0" w="full">
          <Box
            aria-label={listLabel}
            id={listboxId}
            minW="0"
            position="relative"
            role="listbox"
            style={listboxStyle}
            w="full"
          >
            {toRowRuns(virtualItems, rows).map((run) => {
              if (run.kind === 'header') {
                const row = rows[run.item.index];

                return row?.kind === 'header' ? (
                  <PickerGroupHeader
                    key={run.item.key}
                    group={row.group}
                    index={run.item.index}
                    measureElement={measureElement}
                    pinnedShift={run.item.index === pinnedIndex ? pinnedShift : null}
                    start={run.item.start}
                  />
                ) : null;
              }

              // Group wrappers hold only absolutely placed rows, so they take no space in the flow.
              return showHeaders ? (
                <div key={`group:${run.group.id}`} aria-label={run.group.name} role="group">
                  {run.items.map(renderOptionRow)}
                </div>
              ) : (
                <Fragment key={`group:${run.group.id}`}>{run.items.map(renderOptionRow)}</Fragment>
              );
            })}
          </Box>
        </ScrollArea.Content>
      </ScrollArea.Viewport>
      <ScrollArea.Scrollbar>
        <ScrollArea.Thumb />
      </ScrollArea.Scrollbar>
    </ScrollArea.Root>
  );
};

const getRailColor = (group: { colorPalette?: string }): string =>
  group.colorPalette ? `${group.colorPalette}.solid` : 'border.emphasized';

/**
 * Visual only: the enclosing group carries the name for assistive technology. The pinned header stays in flow as
 * the list's only non-absolute row, so `sticky` holds it at the top until the next header pushes it out.
 */
const PickerGroupHeader = <T,>({
  group,
  index,
  measureElement,
  pinnedShift,
  start,
}: {
  group: PickerGroup<T>;
  index: number;
  measureElement: (node: Element | null) => void;
  pinnedShift: number | null;
  start: number;
}) => {
  const railColor = getRailColor(group);
  const style = useMemo<CSSProperties>(
    () =>
      pinnedShift === null
        ? { ...ROW_STYLE, transform: `translateY(${start}px)` }
        : { ...PINNED_HEADER_STYLE, transform: `translateY(${pinnedShift}px)` },
    [pinnedShift, start]
  );
  const countLabel = group.getCountLabel?.(group.options.length);

  return (
    <HStack
      ref={measureElement}
      aria-hidden
      bg="bg.panel"
      borderInlineStartColor={railColor}
      borderInlineStartWidth="2px"
      data-index={index}
      gap="2"
      minW="0"
      pe="3"
      ps="2"
      py="1"
      style={style}
      w="full"
    >
      <Text color={railColor} css={dropdownGroupLabel} minW="0" truncate>
        {group.name}
      </Text>
      <Spacer />
      {countLabel ? (
        <Text color="fg.subtle" flexShrink={0} fontSize="xs">
          {countLabel}
        </Text>
      ) : null}
    </HStack>
  );
};

const PickerOptionRow = <T,>({
  domId,
  id,
  index,
  isActive,
  isCompact,
  isDisabled,
  isSelected,
  measureElement,
  option,
  positionInSet,
  railColor,
  renderOption,
  setSize,
  start,
  onActivate,
  onSelect,
}: {
  domId: string;
  id: string;
  index: number;
  isActive: boolean;
  isCompact: boolean;
  isDisabled: boolean;
  isSelected: boolean;
  measureElement: (node: Element | null) => void;
  option: T;
  positionInSet: number;
  railColor: string | undefined;
  renderOption: (option: T, state: PickerOptionState) => ReactNode;
  setSize: number;
  start: number;
  onActivate: (id: string) => void;
  onSelect: (option: T) => void;
}) => {
  const handleClick = useCallback(() => {
    if (!isDisabled) {
      onSelect(option);
    }
  }, [isDisabled, onSelect, option]);
  const handlePointerMove = useCallback(() => {
    if (!isDisabled) {
      onActivate(id);
    }
  }, [id, isDisabled, onActivate]);
  const style = useMemo<CSSProperties>(() => ({ ...ROW_STYLE, transform: `translateY(${start}px)` }), [start]);

  return (
    // Not focusable: focus stays in the search box, which names the active option and owns keyboard selection.
    <Box
      ref={measureElement}
      aria-disabled={isDisabled || undefined}
      aria-posinset={positionInSet}
      aria-selected={isSelected}
      aria-setsize={setSize}
      bg={isActive && !isDisabled ? 'bg.hover' : undefined}
      borderInlineStartColor={railColor}
      borderInlineStartWidth={railColor ? '2px' : undefined}
      cursor={isDisabled ? 'not-allowed' : undefined}
      data-active={isActive ? '' : undefined}
      data-index={index}
      id={domId}
      minW="0"
      opacity={isDisabled ? 0.4 : undefined}
      px="2"
      py={isCompact ? '0.5' : '1'}
      role="option"
      style={style}
      onClick={handleClick}
      onMouseDown={keepSearchFocus}
      onPointerMove={handlePointerMove}
    >
      {renderOption(option, { isActive, isCompact, isSelected })}
    </Box>
  );
};
