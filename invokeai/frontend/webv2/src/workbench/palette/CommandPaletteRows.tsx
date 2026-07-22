import type { ReactNode, Ref } from 'react';

import { Box, HStack, Icon, Kbd, ScrollArea, Spacer, Text, chakra } from '@chakra-ui/react';
import { dropdownGroupLabel } from '@theme/recipes';
import { ArrowUpRightIcon, CheckIcon } from 'lucide-react';
import { useCallback, useImperativeHandle, useState } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';

import type { PaletteEntry, PaletteRow } from './entries';

const RESULT_LIST_ID = 'command-palette-results';
const LABEL_ROW_HEIGHT_PX = 26;
const ENTRY_ROW_HEIGHT_PX = 40;
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

export const getCommandPaletteRowDomId = (rowId: string): string => `${RESULT_LIST_ID}-${encodeURIComponent(rowId)}`;

const preventFocusSteal = (event: { preventDefault: () => void }) => event.preventDefault();

const SECONDARY_BUTTON_HOVER_STYLE = { color: 'fg' };

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

// Rows are divs, not buttons: they stay out of the tab order (combobox
// aria-activedescendant pattern) and may nest a real secondary-action button.
const rowButtonProps = {
  cursor: 'pointer',
  gap: '2.5',
  h: `${ENTRY_ROW_HEIGHT_PX}px`,
  px: '3',
  rounded: 'sm',
  textAlign: 'start',
  w: 'full',
} as const;

const EntryRow = ({
  domId,
  entry,
  isActive,
  matchIndexes,
  onActive,
  onRun,
  onRunSecondary,
}: {
  domId: string;
  entry: PaletteEntry;
  isActive: boolean;
  matchIndexes?: readonly number[];
  onActive: () => void;
  onRun: () => void;
  onRunSecondary: () => void;
}) => {
  const onSecondaryClick = useCallback(
    (event: React.MouseEvent) => {
      event.stopPropagation();
      onRunSecondary();
    },
    [onRunSecondary]
  );

  return (
    <HStack
      {...rowButtonProps}
      id={domId}
      aria-selected={isActive}
      bg={isActive ? 'bg.emphasized' : undefined}
      role="option"
      onClick={onRun}
      onMouseDown={preventFocusSteal}
      onMouseMove={onActive}
    >
      {entry.thumbnailUrl ? (
        <chakra.img alt="" boxSize="7" flexShrink={0} objectFit="cover" rounded="sm" src={entry.thumbnailUrl} />
      ) : null}
      <Text fontSize="sm" truncate>
        {renderTitle(entry.title, matchIndexes)}
      </Text>
      {entry.subtitle === 'Current' ? (
        <Icon as={CheckIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
      ) : entry.subtitle ? (
        <Text color="fg.subtle" flexShrink={0} fontSize="xs" maxW="45%" truncate>
          {entry.subtitle}
        </Text>
      ) : null}
      <Spacer />
      {entry.secondary && isActive ? (
        <chakra.button
          aria-label={entry.secondary.label}
          color="fg.muted"
          cursor="pointer"
          display="inline-flex"
          flexShrink={0}
          p="1"
          rounded="sm"
          tabIndex={-1}
          title={entry.secondary.label}
          type="button"
          _hover={SECONDARY_BUTTON_HOVER_STYLE}
          onClick={onSecondaryClick}
          onMouseDown={preventFocusSteal}
        >
          <Icon as={ArrowUpRightIcon} boxSize="3.5" />
        </chakra.button>
      ) : null}
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
};

const ScopeRow = ({
  domId,
  isActive,
  label,
  onActive,
  onRun,
}: {
  domId: string;
  isActive: boolean;
  label: string;
  onActive: () => void;
  onRun: () => void;
}) => (
  <HStack
    {...rowButtonProps}
    id={domId}
    aria-selected={isActive}
    bg={isActive ? 'bg.emphasized' : undefined}
    role="option"
    onClick={onRun}
    onMouseDown={preventFocusSteal}
    onMouseMove={onActive}
  >
    <Text color="fg.muted" fontSize="sm" truncate>
      {label}
    </Text>
    <Spacer />
    <Kbd size="sm" textTransform="lowercase">
      tab
    </Kbd>
  </HStack>
);

const VirtualPaletteRow = ({
  activeRowId,
  measureElement,
  row,
  virtualIndex,
  virtualStart,
  onActive,
  onRun,
  onRunSecondary,
}: {
  activeRowId: string | null;
  measureElement: (node: Element | null) => void;
  row: PaletteRow;
  virtualIndex: number;
  virtualStart: number;
  onActive: (rowId: string) => void;
  onRun: (row: PaletteRow) => void;
  onRunSecondary: (row: PaletteRow) => void;
}) => {
  const handleActive = useCallback(() => onActive(row.id), [onActive, row.id]);
  const handleRun = useCallback(() => onRun(row), [onRun, row]);
  const handleRunSecondary = useCallback(() => onRunSecondary(row), [onRunSecondary, row]);
  const isActive = row.id === activeRowId;

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
      ) : row.kind === 'scope' ? (
        <ScopeRow
          domId={getCommandPaletteRowDomId(row.id)}
          isActive={isActive}
          label={row.label}
          onActive={handleActive}
          onRun={handleRun}
        />
      ) : (
        <EntryRow
          domId={getCommandPaletteRowDomId(row.id)}
          entry={row.entry}
          isActive={isActive}
          matchIndexes={row.matchIndexes}
          onActive={handleActive}
          onRun={handleRun}
          onRunSecondary={handleRunSecondary}
        />
      )}
    </Box>
  );
};

export interface CommandPaletteRowsHandle {
  scrollToIndex: (index: number) => void;
}

export const CommandPaletteRows = ({
  activeRowId,
  ref,
  rows,
  onActive,
  onRun,
  onRunSecondary,
}: {
  activeRowId: string | null;
  ref?: Ref<CommandPaletteRowsHandle>;
  rows: PaletteRow[];
  onActive: (rowId: string) => void;
  onRun: (row: PaletteRow) => void;
  onRunSecondary: (row: PaletteRow) => void;
}) => {
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
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

  useImperativeHandle(ref, () => ({ scrollToIndex: (index) => virtualizer.scrollToIndex(index, { align: 'auto' }) }), [
    virtualizer,
  ]);

  return (
    <ScrollArea.Root maxH="min(400px, 55dvh)" size="xs" variant="hover" w="full">
      <ScrollArea.Viewport ref={setScrollElement} aria-label="Command palette results" maxH="inherit" w="full">
        <ScrollArea.Content id={RESULT_LIST_ID} pb="1.5" role="listbox" w="full">
          <Box h={`${virtualizer.totalSize}px`} position="relative" w="full">
            {virtualizer.virtualItems.map((virtualRow) => {
              const row = rows[virtualRow.index];

              return row ? (
                <VirtualPaletteRow
                  key={virtualRow.key}
                  activeRowId={activeRowId}
                  measureElement={virtualizer.measureElement}
                  row={row}
                  virtualIndex={virtualRow.index}
                  virtualStart={virtualRow.start}
                  onActive={onActive}
                  onRun={onRun}
                  onRunSecondary={onRunSecondary}
                />
              ) : null;
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
