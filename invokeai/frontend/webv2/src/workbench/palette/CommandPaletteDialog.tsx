import type { ChangeEvent, KeyboardEvent, ReactNode } from 'react';

import { Box, Dialog, HStack, Icon, Kbd, Portal, ScrollArea, Spacer, Text, chakra } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Button } from '@platform/ui';
import { EmptyState } from '@platform/ui/EmptyState';
import { useQueries } from '@tanstack/react-query';
import { dropdownGroupLabel } from '@theme/recipes';
import { formatHotkeyForPlatform } from '@workbench/hotkeys/keys';
import { registerHotkeyModalLayer } from '@workbench/hotkeys/modalLayer';
import { CheckIcon, SearchIcon } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { useVirtualizer } from 'react-hook-tanstack-virtual';

import type { PaletteEntry, PaletteRow, PaletteSearchProvider, PaletteStage, ProviderResultSection } from './entries';

import {
  buildProviderSectionRows,
  buildScopeRows,
  buildStageEntries,
  PROVIDER_MIN_QUERY_LENGTH,
  searchPaletteRows,
} from './entries';
import { getRecentEntryIds, recordRecentEntry } from './recents';

/**
 * The palette surface: a top-anchored quiet overlay with one input, one mixed
 * virtualized list (fixed section order, fuzzy-ranked within sections), and a
 * footer hint bar. Follows the AddNodeDialog combobox pattern — focus stays on
 * the input, the highlighted row travels via aria-activedescendant, and hover
 * only claims the highlight on real mouse movement (not scroll-under-cursor).
 *
 * Two overlays stack on the root list: a value-picker *stage* (enum settings)
 * and a provider *scope* (deep entity search). Both render as a chip in the
 * input; Esc or Backspace-on-empty pops them before anything closes.
 */

const RESULT_LIST_ID = 'command-palette-results';
const LABEL_ROW_HEIGHT_PX = 26;
const ENTRY_ROW_HEIGHT_PX = 40;
const PAGE_JUMP_ROWS = 8;
const PROVIDER_DEBOUNCE_MS = 200;
const VIRTUALIZER_INITIAL_RECT = { height: 384, width: 0 };

const INPUT_PLACEHOLDER_STYLE = { color: 'fg.subtle' };
const NO_PROVIDERS: PaletteSearchProvider[] = [];

const NAV_HINT_KEYS = ['↑', '↓'];
const ENTER_HINT_KEYS = ['↵'];
const ESC_HINT_KEYS = ['esc'];
const TAB_HINT_KEYS = ['tab'];
const MOD_ENTER_HINT_KEYS = [formatHotkeyForPlatform('mod')[0] ?? 'ctrl', '↵'];

const getRowDomId = (index: number): string => `${RESULT_LIST_ID}-${index}`;

/** Rows never take focus — mousedown is swallowed so the input keeps the caret through clicks. */
const preventFocusSteal = (event: { preventDefault: () => void }) => event.preventDefault();

const useDebouncedValue = <Value,>(value: Value, delayMs: number): Value => {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    if (Object.is(debounced, value)) {
      return;
    }

    const timeoutId = window.setTimeout(() => setDebounced(value), delayMs);

    return () => window.clearTimeout(timeoutId);
  }, [debounced, delayMs, value]);

  return debounced;
};

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

const rowButtonProps = {
  as: 'button',
  cursor: 'pointer',
  gap: '2.5',
  h: `${ENTRY_ROW_HEIGHT_PX}px`,
  px: '3',
  rounded: 'sm',
  tabIndex: -1,
  textAlign: 'start',
  w: 'full',
} as const;

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
    {...rowButtonProps}
    id={id}
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

const ScopeRow = ({
  id,
  isActive,
  label,
  onActive,
  onRun,
}: {
  id: string;
  isActive: boolean;
  label: string;
  onActive: () => void;
  onRun: () => void;
}) => (
  <HStack
    {...rowButtonProps}
    id={id}
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

  const isActive = virtualIndex === activeIndex;

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
          id={getRowDomId(virtualIndex)}
          isActive={isActive}
          label={row.label}
          onActive={onActive}
          onRun={onRun}
        />
      ) : (
        <EntryRow
          id={getRowDomId(virtualIndex)}
          entry={row.entry}
          isActive={isActive}
          matchIndexes={row.matchIndexes}
          onActive={onActive}
          onRun={onRun}
        />
      )}
    </Box>
  );
};

const FooterHint = ({ children, keys }: { children: string; keys: string[] }) => (
  <HStack gap="1">
    {keys.map((key) => (
      <Kbd key={key} size="sm" textTransform="lowercase">
        {key}
      </Kbd>
    ))}
    <Text>{children}</Text>
  </HStack>
);

export const CommandPaletteDialog = ({
  entries,
  isOpen,
  onClose,
  providers = NO_PROVIDERS,
}: {
  entries: PaletteEntry[];
  isOpen: boolean;
  onClose: () => void;
  providers?: PaletteSearchProvider[];
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
    <Dialog.Root
      closeOnEscape={false}
      lazyMount
      open={isOpen}
      scrollBehavior="inside"
      unmountOnExit
      onOpenChange={onDialogOpenChange}
    >
      {isOpen ? <CommandPaletteContent entries={entries} providers={providers} onClose={onClose} /> : null}
    </Dialog.Root>
  );
};

const CommandPaletteContent = ({
  entries,
  onClose,
  providers,
}: {
  entries: PaletteEntry[];
  onClose: () => void;
  providers: PaletteSearchProvider[];
}) => {
  const [query, setQuery] = useState('');
  const [activeIndex, setActiveIndex] = useState(-1);
  const [stage, setStage] = useState<PaletteStage | null>(null);
  const [scopeProviderId, setScopeProviderId] = useState<string | null>(null);
  // Snapshot at open: recents should not reshuffle while the palette is up.
  const [recentIds] = useState(getRecentEntryIds);
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(null);
  const [inputElement, setInputElement] = useState<HTMLInputElement | null>(null);

  // Keyboard flow lives in the input; overlay transitions must never leave
  // focus stranded on an unmounted row or the retry button.
  useEffect(() => {
    inputElement?.focus();
  }, [inputElement, scopeProviderId, stage]);

  useMountEffect(() => registerHotkeyModalLayer('command-palette'));

  const scopeProvider = scopeProviderId ? (providers.find((p) => p.id === scopeProviderId) ?? null) : null;
  // '>' at position 0 is a courtesy alias for the Commands scope.
  const isCommandsScope = !stage && !scopeProvider && query.startsWith('>');
  const localQuery = isCommandsScope ? query.slice(1) : query;
  const trimmedQuery = localQuery.trim();

  const resetHighlight = useCallback(() => setActiveIndex(-1), []);

  const popOverlay = useCallback(() => {
    setStage(null);
    setScopeProviderId(null);
    setQuery('');
    setActiveIndex(-1);
  }, []);

  // --- Async entity providers -------------------------------------------
  const debouncedQuery = useDebouncedValue(trimmedQuery, PROVIDER_DEBOUNCE_MS);
  const shouldSearchProviders = debouncedQuery.length >= PROVIDER_MIN_QUERY_LENGTH && !stage && !isCommandsScope;
  const activeProviders = useMemo(
    () => (stage || isCommandsScope ? NO_PROVIDERS : scopeProvider ? [scopeProvider] : providers),
    [isCommandsScope, providers, scopeProvider, stage]
  );
  // `combine` output is structurally shared by TanStack Query, so its
  // identity is stable across renders that change nothing — safe memo input.
  const providerResults = useQueries({
    combine: (results) =>
      results.map((result) => ({
        data: result.data,
        isError: result.isError,
        isFetching: result.isFetching,
        refetch: result.refetch,
      })),
    queries: activeProviders.map((provider) => ({
      enabled: shouldSearchProviders,
      queryFn: () => Promise.resolve(provider.search(debouncedQuery)),
      queryKey: ['command-palette', provider.id, debouncedQuery],
      retry: false,
      staleTime: 30_000,
    })),
  });
  const providerSections = useMemo<ProviderResultSection[]>(
    () =>
      activeProviders.map((provider, index) => ({
        entries: providerResults[index]?.data ?? [],
        isFetching: shouldSearchProviders && (providerResults[index]?.isFetching ?? false),
        provider,
      })),
    [activeProviders, providerResults, shouldSearchProviders]
  );
  const scopedQueryResult = scopeProvider ? providerResults[0] : undefined;

  // --- Row assembly ------------------------------------------------------
  const onStageApplied = useCallback(() => popOverlay(), [popOverlay]);

  const rows = useMemo<PaletteRow[]>(() => {
    if (stage) {
      return searchPaletteRows(buildStageEntries(stage, onStageApplied), query, [], { showAllOnEmpty: true });
    }

    if (scopeProvider) {
      return buildProviderSectionRows(providerSections, null);
    }

    const localRows = searchPaletteRows(entries, localQuery, recentIds, { commandsOnly: isCommandsScope });

    if (trimmedQuery.length === 0 || isCommandsScope) {
      return localRows;
    }

    return [
      ...localRows,
      ...(trimmedQuery.length >= PROVIDER_MIN_QUERY_LENGTH ? buildProviderSectionRows(providerSections) : []),
      ...buildScopeRows(providers, trimmedQuery),
    ];
  }, [
    entries,
    isCommandsScope,
    localQuery,
    onStageApplied,
    providers,
    providerSections,
    query,
    recentIds,
    scopeProvider,
    stage,
    trimmedQuery,
  ]);
  const navigableRowIndexes = useMemo(
    () => rows.flatMap((row, index) => (row.kind === 'label' ? [] : [index])),
    [rows]
  );
  // The highlight always lands on a navigable row; -1 (or a stale index after
  // a query change) resolves to the first hit so Enter runs the top result.
  const effectiveActiveIndex = navigableRowIndexes.includes(activeIndex) ? activeIndex : (navigableRowIndexes[0] ?? -1);
  const activeRow = effectiveActiveIndex === -1 ? undefined : rows[effectiveActiveIndex];

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

  // --- Execution ---------------------------------------------------------
  const enterScope = useCallback((providerId: string) => {
    setScopeProviderId(providerId);
    setActiveIndex(-1);
  }, []);

  const runRow = useCallback(
    (row: PaletteRow) => {
      if (row.kind === 'label') {
        return;
      }

      if (row.kind === 'scope') {
        enterScope(row.providerId);
        return;
      }

      recordRecentEntry(row.entry.id);

      if (row.entry.stage) {
        setStage(row.entry.stage);
        setQuery('');
        setActiveIndex(-1);
        return;
      }

      if (row.entry.keepOpen) {
        void row.entry.run();
        return;
      }

      // Close first so command side effects (focus, navigation, widget
      // placement) land in a palette-free world.
      onClose();
      void row.entry.run();
    },
    [enterScope, onClose]
  );

  const runSecondary = useCallback(
    (entry: PaletteEntry) => {
      if (!entry.secondary) {
        return;
      }

      recordRecentEntry(entry.id);
      onClose();
      void entry.secondary.run();
    },
    [onClose]
  );

  const moveActive = useCallback(
    (offset: number) => {
      if (navigableRowIndexes.length === 0) {
        return;
      }

      const position = navigableRowIndexes.indexOf(effectiveActiveIndex);
      const nextPosition =
        Math.abs(offset) === 1
          ? (position + offset + navigableRowIndexes.length) % navigableRowIndexes.length
          : Math.min(Math.max(position + offset, 0), navigableRowIndexes.length - 1);
      const nextIndex = navigableRowIndexes[nextPosition];

      if (nextIndex === undefined) {
        return;
      }

      setActiveIndex(nextIndex);
      virtualizer.scrollToIndex(nextIndex, { align: 'auto' });
    },
    [effectiveActiveIndex, navigableRowIndexes, virtualizer]
  );

  // The dialog owns Esc entirely (closeOnEscape is off so Ark's dismiss
  // listener cannot race the overlay pop): pop the stage/scope first, close
  // only from the root state.
  const onEscape = useCallback(() => {
    if (stage || scopeProviderId) {
      popOverlay();
      return;
    }

    onClose();
  }, [onClose, popOverlay, scopeProviderId, stage]);

  const onContentKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        onEscape();
      }
    },
    [onEscape]
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
      } else if (event.key === 'Tab') {
        event.preventDefault();

        if (activeRow?.kind === 'scope') {
          enterScope(activeRow.providerId);
        }
      } else if (event.key === 'Backspace' && query.length === 0 && (stage || scopeProviderId)) {
        event.preventDefault();
        popOverlay();
      } else if (event.key === 'Enter') {
        event.preventDefault();

        if (!activeRow) {
          return;
        }

        if ((event.metaKey || event.ctrlKey) && activeRow.kind === 'entry' && activeRow.entry.secondary) {
          runSecondary(activeRow.entry);
          return;
        }

        runRow(activeRow);
      }
    },
    [activeRow, enterScope, moveActive, popOverlay, query.length, runRow, runSecondary, scopeProviderId, stage]
  );

  const onSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setQuery(event.currentTarget.value);
      resetHighlight();
    },
    [resetHighlight]
  );

  // --- Rendering ---------------------------------------------------------
  const retryScopedSearch = useCallback(() => void scopedQueryResult?.refetch(), [scopedQueryResult]);

  const chipLabel = stage ? stage.title : (scopeProvider?.label ?? null);
  const placeholder = stage
    ? 'Pick a value…'
    : scopeProvider
      ? `Search ${scopeProvider.label.toLowerCase()}…`
      : 'Search commands, settings, and more…';

  let emptyState: ReactNode = null;

  if (rows.length === 0) {
    if (scopeProvider && scopedQueryResult?.isError) {
      emptyState = (
        <EmptyState danger py="6" title={`Couldn't search ${scopeProvider.label.toLowerCase()}`}>
          <Button size="xs" variant="subtle" onClick={retryScopedSearch}>
            Retry
          </Button>
        </EmptyState>
      );
    } else if (scopeProvider && trimmedQuery.length < PROVIDER_MIN_QUERY_LENGTH) {
      emptyState = <EmptyState py="6" title={`Keep typing to search ${scopeProvider.label.toLowerCase()}`} />;
    } else if (scopeProvider && scopedQueryResult?.isFetching) {
      emptyState = <EmptyState py="6" title="Searching…" />;
    } else {
      emptyState = <EmptyState py="6" title={`No results for “${trimmedQuery}”`} />;
    }
  }

  const hasScopeRows = rows.some((row) => row.kind === 'scope');
  const secondaryHint = activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined;

  return (
    <Portal>
      <Dialog.Backdrop bg="blackAlpha.300" />
      <Dialog.Positioner alignItems="flex-start" pt="15vh">
        <Dialog.Content
          maxW="none"
          overflow="hidden"
          p="0"
          w="min(560px, calc(100vw - 32px))"
          onKeyDown={onContentKeyDown}
        >
          <HStack borderBottomWidth="1px" borderColor="border.emphasized" flexShrink={0} gap="2.5" h="12" px="3">
            <Icon as={SearchIcon} boxSize="4" color="fg.muted" flexShrink={0} />
            {chipLabel ? (
              <Text
                bg="bg.emphasized"
                borderRadius="sm"
                color="fg"
                flexShrink={0}
                fontSize="xs"
                fontWeight="600"
                px="1.5"
                py="0.5"
              >
                {chipLabel}
              </Text>
            ) : null}
            <chakra.input
              ref={setInputElement}
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
              placeholder={placeholder}
              role="combobox"
              value={query}
              w="full"
              _placeholder={INPUT_PLACEHOLDER_STYLE}
              onChange={onSearchChange}
              onKeyDown={onSearchKeyDown}
            />
          </HStack>
          {rows.length === 0 ? (
            emptyState
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
            <FooterHint keys={NAV_HINT_KEYS}>Navigate</FooterHint>
            <FooterHint keys={ENTER_HINT_KEYS}>{stage ? 'Pick' : 'Run'}</FooterHint>
            {secondaryHint ? <FooterHint keys={MOD_ENTER_HINT_KEYS}>{secondaryHint}</FooterHint> : null}
            {hasScopeRows || activeRow?.kind === 'scope' ? <FooterHint keys={TAB_HINT_KEYS}>Scope</FooterHint> : null}
            <Spacer />
            <FooterHint keys={ESC_HINT_KEYS}>{stage || scopeProviderId ? 'Back' : 'Close'}</FooterHint>
          </HStack>
        </Dialog.Content>
      </Dialog.Positioner>
    </Portal>
  );
};
