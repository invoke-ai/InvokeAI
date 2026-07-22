import type { ChangeEvent, KeyboardEvent, ReactNode } from 'react';

import { Box, Dialog, HStack, Icon, Kbd, Portal, ScrollArea, Spacer, Text, chakra } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { EmptyState } from '@platform/ui/EmptyState';
import { dropdownGroupLabel } from '@theme/recipes';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { SearchIcon } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';

import type { PaletteEntry, PaletteRow } from './entries';

import { searchPaletteRows } from './entries';
import { getRecentEntryIds, recordRecentEntry } from './recents';

/**
 * The palette surface: a top-anchored quiet overlay with one input, one mixed
 * virtualized list (fixed section order, fuzzy-ranked within sections), and a
 * footer hint bar. Follows the AddNodeDialog combobox pattern — focus stays on
 * the input, the highlighted row travels via aria-activedescendant, and hover
 * only claims the highlight on real mouse movement (not scroll-under-cursor).
 */

const RESULT_LIST_ID = 'command-palette-results';
const LABEL_ROW_HEIGHT_PX = 26;
const ENTRY_ROW_HEIGHT_PX = 40;
const PAGE_JUMP_ROWS = 8;
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

const INPUT_PLACEHOLDER_STYLE = { color: 'fg.subtle' };

const getRowDomId = (index: number): string => `${RESULT_LIST_ID}-${index}`;

/** Render the entry title with fuzzysort's matched characters subtly emphasized. */
const renderTitle = (title: string, matchIndexes?: readonly number[]): ReactNode => {
  if (!matchIndexes || matchIndexes.length === 0) {
    return title;
  }

  const matched = new Set(matchIndexes);
  const segments: ReactNode[] = [];
  let start = 0;

  for (let index = 1; index <= title.length; index += 1) {
    if (index === title.length || matched.has(index) !== matched.has(start)) {
      const text = title.slice(start, index);

      segments.push(
        matched.has(start) ? (
          <Text key={start} as="span" fontWeight="600">
            {text}
          </Text>
        ) : (
          text
        )
      );
      start = index;
    }
  }

  return segments;
};

const EntryRow = ({
  id,
  isActive,
  matchIndexes,
  entry,
  onActive,
  onRun,
}: {
  id: string;
  isActive: boolean;
  matchIndexes?: readonly number[];
  entry: PaletteEntry;
  onActive: () => void;
  onRun: () => void;
}) => (
  <HStack
    id={id}
    as="button"
    aria-selected={isActive}
    bg={isActive ? 'bg.emphasized' : undefined}
    role="option"
    tabIndex={-1}
    cursor="pointer"
    gap="2.5"
    h={`${ENTRY_ROW_HEIGHT_PX}px`}
    px="3"
    rounded="sm"
    textAlign="start"
    w="full"
    onClick={onRun}
    onMouseMove={onActive}
  >
    <Text fontSize="sm" truncate>
      {renderTitle(entry.title, matchIndexes)}
    </Text>
    {entry.subtitle ? (
      <Text color="fg.subtle" flexShrink={0} fontSize="xs">
        {entry.subtitle}
      </Text>
    ) : null}
    <Spacer />
    {entry.keys ? (
      <HStack flexShrink={0} gap="0.5">
        {entry.keys.map((part) => (
          <Kbd key={part} size="sm" textTransform="lowercase">
            {part}
          </Kbd>
        ))}
      </HStack>
    ) : null}
  </HStack>
);

const VirtualPaletteRow = ({
  activeIndex,
  measureElement,
  rows,
  virtualIndex,
  virtualStart,
  onActiveIndexChange,
  onRunRow,
}: {
  activeIndex: number;
  measureElement: (node: Element | null) => void;
  rows: PaletteRow[];
  virtualIndex: number;
  virtualStart: number;
  onActiveIndexChange: (index: number) => void;
  onRunRow: (row: PaletteRow) => void;
}) => {
  const row = rows[virtualIndex];
  const onActive = useCallback(() => onActiveIndexChange(virtualIndex), [onActiveIndexChange, virtualIndex]);
  const onRun = useCallback(() => row && onRunRow(row), [onRunRow, row]);

  if (!row) {
    return null;
  }

  return (
    <Box
      ref={measureElement}
      data-index={virtualIndex}
      left="0"
      position="absolute"
      top="0"
      transform={`translateY(${virtualStart}px)`}
      w="full"
    >
      {row.kind === 'label' ? (
        <Text css={dropdownGroupLabel} pb="1" pt="2" px="3" role="presentation">
          {row.label}
        </Text>
      ) : (
        <EntryRow
          id={getRowDomId(virtualIndex)}
          entry={row.entry}
          isActive={virtualIndex === activeIndex}
          matchIndexes={row.matchIndexes}
          onActive={onActive}
          onRun={onRun}
        />
      )}
    </Box>
  );
};

export const CommandPaletteDialog = ({
  entries,
  isOpen,
  onClose,
}: {
  entries: PaletteEntry[];
  isOpen: boolean;
  onClose: () => void;
}) => {
  const onDialogOpenChange = useCallback(
    (event: { open: boolean }) => {
      if (!event.open) {
        onClose();
      }
    },
    [onClose]
  );

  return (
    <Dialog.Root lazyMount open={isOpen} scrollBehavior="inside" unmountOnExit onOpenChange={onDialogOpenChange}>
      {isOpen ? <CommandPaletteContent entries={entries} onClose={onClose} /> : null}
    </Dialog.Root>
  );
};

const CommandPaletteContent = ({ entries, onClose }: { entries: PaletteEntry[]; onClose: () => void }) => {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(-1);
  // Snapshot at open: recents should not reshuffle while the palette is up.
  const [recentIds] = useState(getRecentEntryIds);
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);

  useMountEffect(() => registerHotkeyModalLayer('command-palette'));

  const rows = useMemo(() => searchPaletteRows(entries, query, recentIds), [entries, query, recentIds]);
  const entryRowIndexes = useMemo(() => rows.flatMap((row, index) => (row.kind === 'entry' ? [index] : [])), [rows]);
  // The highlight always lands on an entry row; -1 (or a stale index after a
  // query change) resolves to the first hit so Enter runs the top result.
  const effectiveActiveIndex = entryRowIndexes.includes(activeIndex) ? activeIndex : (entryRowIndexes[0] ?? -1);

  const estimateRowSize = useCallback(
    (index: number) => (rows[index]?.kind === 'label' ? LABEL_ROW_HEIGHT_PX : ENTRY_ROW_HEIGHT_PX),
    [rows]
  );
  const getRowKey = useCallback((index: number) => rows[index]?.id ?? index, [rows]);
  const getScrollElement = useCallback(() => scrollElement, [scrollElement]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: estimateRowSize,
    getItemKey: getRowKey,
    getScrollElement,
    initialRect: VIRTUALIZER_INITIAL_RECT,
    overscan: 8,
  });

  const runRow = useCallback(
    (row: PaletteRow) => {
      if (row.kind !== 'entry') {
        return;
      }

      recordRecentEntry(row.entry.id);

      if (row.entry.keepOpen) {
        void row.entry.run();
        return;
      }

      // Close first so command side effects (focus, navigation, widget
      // placement) land in a palette-free world.
      onClose();
      void row.entry.run();
    },
    [onClose]
  );

  const moveActive = useCallback(
    (offset: number) => {
      if (entryRowIndexes.length === 0) {
        return;
      }

      const position = entryRowIndexes.indexOf(effectiveActiveIndex);
      const nextPosition =
        Math.abs(offset) === 1
          ? (position + offset + entryRowIndexes.length) % entryRowIndexes.length
          : Math.min(Math.max(position + offset, 0), entryRowIndexes.length - 1);
      const nextIndex = entryRowIndexes[nextPosition];

      if (nextIndex === undefined) {
        return;
      }

      setActiveIndex(nextIndex);
      virtualizer.scrollToIndex(nextIndex, { align: 'auto' });
    },
    [effectiveActiveIndex, entryRowIndexes, virtualizer]
  );

  const onSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        moveActive(1);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        moveActive(-1);
      } else if (event.key === 'PageDown') {
        event.preventDefault();
        moveActive(PAGE_JUMP_ROWS);
      } else if (event.key === 'PageUp') {
        event.preventDefault();
        moveActive(-PAGE_JUMP_ROWS);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        const row = effectiveActiveIndex === -1 ? undefined : rows[effectiveActiveIndex];

        if (row) {
          runRow(row);
        }
      }
    },
    [effectiveActiveIndex, moveActive, rows, runRow]
  );

  const onSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setQuery(event.currentTarget.value);
    setActiveIndex(-1);
  }, []);

  return (
    <Portal>
      <Dialog.Backdrop bg="blackAlpha.300" />
      <Dialog.Positioner alignItems="flex-start" pt="15vh">
        <Dialog.Content maxW="none" overflow="hidden" p="0" w="min(560px, calc(100vw - 32px))">
          <HStack borderBottomWidth="1px" borderColor="border.emphasized" flexShrink={0} gap="2.5" h="12" px="3">
            <Icon as={SearchIcon} boxSize="4" color="fg.muted" flexShrink={0} />
            <chakra.input
              autoFocus
              aria-activedescendant={effectiveActiveIndex === -1 ? undefined : getRowDomId(effectiveActiveIndex)}
              aria-controls={RESULT_LIST_ID}
              aria-expanded="true"
              aria-label="Search commands and settings"
              bg="transparent"
              color="fg"
              flex="1"
              fontSize="sm"
              outline="none"
              placeholder="Search commands, settings, and more…"
              role="combobox"
              value={query}
              w="full"
              _placeholder={INPUT_PLACEHOLDER_STYLE}
              onChange={onSearchChange}
              onKeyDown={onSearchKeyDown}
            />
          </HStack>
          {rows.length === 0 ? (
            <EmptyState py="6" title={`No results for “${query.trim()}”`} />
          ) : (
            <ScrollArea.Root maxH="min(400px, 55dvh)" size="xs" variant="hover" w="full">
              <ScrollArea.Viewport ref={setScrollElement} aria-label="Command palette results" maxH="inherit" w="full">
                <ScrollArea.Content id={RESULT_LIST_ID} pb="1.5" role="listbox" w="full">
                  <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
                    {virtualizer.virtualItems.map((virtualRow) => (
                      <VirtualPaletteRow
                        key={virtualRow.key}
                        activeIndex={effectiveActiveIndex}
                        measureElement={virtualizer.measureElement}
                        rows={rows}
                        virtualIndex={virtualRow.index}
                        virtualStart={virtualRow.start}
                        onActiveIndexChange={setActiveIndex}
                        onRunRow={runRow}
                      />
                    ))}
                  </Box>
                </ScrollArea.Content>
              </ScrollArea.Viewport>
              <ScrollArea.Scrollbar>
                <ScrollArea.Thumb />
              </ScrollArea.Scrollbar>
            </ScrollArea.Root>
          )}
          <HStack
            borderColor="border.emphasized"
            borderTopWidth="1px"
            color="fg.subtle"
            flexShrink={0}
            fontSize="xs"
            gap="4"
            h="8"
            hideBelow="sm"
            px="3"
          >
            <HStack gap="1">
              <Kbd size="sm">↑</Kbd>
              <Kbd size="sm">↓</Kbd>
              <Text>Navigate</Text>
            </HStack>
            <HStack gap="1">
              <Kbd size="sm">↵</Kbd>
              <Text>Run</Text>
            </HStack>
            <HStack gap="1">
              <Kbd size="sm" textTransform="lowercase">
                esc
              </Kbd>
              <Text>Close</Text>
            </HStack>
          </HStack>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};
