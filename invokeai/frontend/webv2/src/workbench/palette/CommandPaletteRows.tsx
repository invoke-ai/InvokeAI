import type { ReactNode, Ref } from 'react';

import { Box, Command, Icon, Kbd, Spacer, Text, VisuallyHidden, chakra } from '@chakra-ui/react';
import { Scrollable } from '@platform/ui/Scrollable';
import { useVirtualizer } from '@tanstack/react-virtual';
import { dropdownGroupLabel } from '@theme/recipes';
import { ShortcutKeyGlyph, ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { CheckIcon, ExternalLinkIcon } from 'lucide-react';
import { useCallback, useImperativeHandle, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { NavigablePaletteRow, PaletteEntry, PaletteRow } from './entries';

export const COMMAND_PALETTE_RESULTS_ID = 'command-palette-results';
/** Names the results listbox, which the combobox labels by its label id. */
export const COMMAND_PALETTE_RESULTS_LABEL_ID = 'command-palette-results-label';
const LABEL_ROW_HEIGHT_PX = 26;
const ENTRY_ROW_HEIGHT_PX = 40;
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

export const getCommandPaletteRowDomId = (rowId: string): string =>
  `${COMMAND_PALETTE_RESULTS_ID}-${encodeURIComponent(rowId)}`;

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

const rowStyleProps = {
  alignItems: 'center',
  display: 'flex',
  gap: '2.5',
  h: `${ENTRY_ROW_HEIGHT_PX}px`,
  px: '3',
  rounded: 'sm',
  textAlign: 'start',
  w: 'full',
} as const;

/** The combobox only marks chosen values selected, and the palette has none; the highlighted option is the selection. */
const PaletteItem = ({
  children,
  isActive,
  row,
}: {
  children: ReactNode;
  isActive: boolean;
  row: NavigablePaletteRow;
}) => {
  const isError = row.kind === 'provider-error';

  return (
    <Command.Item
      {...rowStyleProps}
      aria-selected={isActive}
      bg={isActive ? (isError ? 'bg.error' : 'bg.hover') : undefined}
      color={isError ? 'fg.error' : undefined}
      item={row}
    >
      {children}
    </Command.Item>
  );
};

const EntryRowContent = ({ entry, matchIndexes }: { entry: PaletteEntry; matchIndexes?: readonly number[] }) => {
  const { t } = useTranslation();

  return (
    <>
      {entry.thumbnailUrl ? (
        <chakra.img alt="" boxSize="7" flexShrink={0} objectFit="cover" rounded="sm" src={entry.thumbnailUrl} />
      ) : null}
      <Command.ItemText fontSize="lg" truncate>
        {renderTitle(entry.title, matchIndexes)}
      </Command.ItemText>
      {entry.isCurrent ? (
        <Icon as={CheckIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
      ) : entry.subtitle ? (
        <Text color="fg.subtle" flexShrink={0} fontSize="md" maxW="45%" truncate>
          {entry.subtitle}
        </Text>
      ) : null}
      <Spacer />
      {entry.keys ? <ShortcutKeycaps parts={entry.keys} /> : null}
      {entry.opensExternally ? (
        <>
          <Icon as={ExternalLinkIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
          <VisuallyHidden>{t('commandPalette.opensInNewTab')}</VisuallyHidden>
        </>
      ) : null}
    </>
  );
};

const VirtualPaletteRow = ({
  activeRowId,
  measureElement,
  row,
  virtualIndex,
  virtualStart,
}: {
  activeRowId: string | null;
  measureElement: (node: Element | null) => void;
  row: PaletteRow;
  virtualIndex: number;
  virtualStart: number;
}) => (
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
      <PaletteItem isActive={row.id === activeRowId} row={row}>
        {row.kind === 'entry' ? (
          <EntryRowContent entry={row.entry} matchIndexes={row.matchIndexes} />
        ) : (
          <>
            <Command.ItemText color={row.kind === 'scope' ? 'fg.muted' : undefined} fontSize="lg" truncate>
              {row.label}
            </Command.ItemText>
            {row.kind === 'scope' ? (
              <>
                <Spacer />
                <Kbd textTransform="lowercase">
                  <ShortcutKeyGlyph fallback="tab" part="tab" />
                </Kbd>
              </>
            ) : null}
          </>
        )}
      </PaletteItem>
    )}
  </Box>
);

export interface CommandPaletteRowsHandle {
  /** Scrolls a row into view; the first actionable row scrolls to the top, bringing its leading labels along. */
  scrollToRow: (rowId: string) => void;
}

export const CommandPaletteRows = ({
  activeRowId,
  isBusy,
  ref,
  rows,
}: {
  activeRowId: string | null;
  isBusy: boolean;
  ref?: Ref<CommandPaletteRowsHandle>;
  rows: PaletteRow[];
}) => {
  const { t } = useTranslation();
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
  const estimateRowSize = useCallback(
    (index: number) => (rows[index]?.kind === 'label' ? LABEL_ROW_HEIGHT_PX : ENTRY_ROW_HEIGHT_PX),
    [rows]
  );
  const getRowKey = useCallback((index: number) => rows[index]?.id ?? index, [rows]);
  const getScrollElement = useCallback(() => scrollElement, [scrollElement]);
  // eslint-disable-next-line react/incompatible-library -- TanStack Virtual exposes imperative functions that the compiler cannot memoize safely.
  const virtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: estimateRowSize,
    getItemKey: getRowKey,
    getScrollElement,
    initialRect: VIRTUALIZER_INITIAL_RECT,
    overscan: 8,
  });

  useImperativeHandle(
    ref,
    () => ({
      scrollToRow: (rowId) => {
        const index = rows.findIndex((row) => row.id === rowId);

        if (index === -1) {
          return;
        }

        const target = rows.slice(0, index).every((row) => row.kind === 'label') ? 0 : index;
        const offset = virtualizer.getOffsetForIndex(target, 'auto')?.[0];

        // A scroll starts a frame loop that re-renders the list; the combobox also asks when nothing needs to move.
        if (offset !== undefined && offset !== (virtualizer.scrollOffset ?? 0)) {
          virtualizer.scrollToIndex(target, { align: 'auto' });
        }
      },
    }),
    [rows, virtualizer]
  );

  return (
    <>
      {/* Hidden from reading order; the listbox still takes its name from it. */}
      <span hidden id={COMMAND_PALETTE_RESULTS_LABEL_ID}>
        {t('commandPalette.resultsLabel')}
      </span>
      <Scrollable maxH="min(400px, 55dvh)" viewportRef={setScrollElement} w="full">
        <Command.List aria-busy={isBusy} pb="1.5">
          <Box h={`${virtualizer.getTotalSize()}px`} position="relative" w="full">
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const row = rows[virtualRow.index];

              return row ? (
                <VirtualPaletteRow
                  key={virtualRow.key}
                  activeRowId={activeRowId}
                  measureElement={virtualizer.measureElement}
                  row={row}
                  virtualIndex={virtualRow.index}
                  virtualStart={virtualRow.start}
                />
              ) : null;
            })}
          </Box>
        </Command.List>
      </Scrollable>
    </>
  );
};
