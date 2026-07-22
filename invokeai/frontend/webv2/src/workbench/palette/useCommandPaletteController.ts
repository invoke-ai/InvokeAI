import type { DateTokenKey } from '@platform/search/dateTokens';
/* eslint-disable react/react-compiler */
import type { ChangeEvent, KeyboardEvent } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import {
  completeTrailingDateToken,
  formatDateRangeLabel,
  isPossibleDatePrefix,
  matchTrailingDateToken,
  parseDateTokens,
} from '@platform/search/dateTokens';
import { useQueries } from '@tanstack/react-query';
import { useCallback, useMemo, useRef, useState } from 'react';

import type { PaletteEntry, PaletteRow, PaletteSearchProvider, PaletteStage, ProviderResultSection } from './entries';

import { getPaletteContributionKey } from './contributionKey';
import {
  buildProviderSectionRows,
  buildScopeRows,
  buildStageEntries,
  PROVIDER_MIN_QUERY_LENGTH,
  resolveActivePaletteRow,
  SEARCH_SCOPE_GROUP,
  searchPaletteRows,
  STAGE_ENTRY_ID_PREFIX,
} from './entries';
import { getPaletteProviderQueryKey } from './providerQueryKey';
import { getRecentEntryIds, recordRecentEntry } from './recents';

const PAGE_JUMP_ROWS = 8;
const PROVIDER_DEBOUNCE_MS = 200;
const NO_PROVIDERS: PaletteSearchProvider[] = [];

/**
 * Completions offered while a date token is being typed. `Nd` resolves to a
 * single date N days ago, so the labels are key-specific: `from:7d` is a
 * starting point ("Past week"), while `date:7d` is one day ("A week ago").
 */
const DATE_TOKEN_SUGGESTIONS: Record<DateTokenKey, ReadonlyArray<{ label: string; value: string }>> = {
  date: [
    { label: 'Today', value: 'today' },
    { label: 'Yesterday', value: 'yesterday' },
    { label: 'A week ago', value: '7d' },
  ],
  from: [
    { label: 'Today', value: 'today' },
    { label: 'Yesterday', value: 'yesterday' },
    { label: 'Past week', value: '7d' },
  ],
  to: [
    { label: 'Today', value: 'today' },
    { label: 'Yesterday', value: 'yesterday' },
  ],
};

const DATE_FORMAT_HINT = 'Or type a date — YYYY-MM-DD, 7d, 2w, 1m';

type ScrollToIndex = (index: number) => void;

export interface CommandPaletteController {
  activeRow: PaletteRow | undefined;
  activeRowId: string | null;
  chipLabel: string | null;
  /** Inline feedback for an unparseable date token value, or null. */
  dateInvalidHint: string | null;
  /** Formatted label of the applied date range, or null. */
  dateSummary: string | null;
  hasScopeRows: boolean;
  isOverlayOpen: boolean;
  onContentKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  /** Pop the active scope/stage back to the root palette (chip dismiss). */
  onPopOverlay: () => void;
  onRetry: () => void;
  onRowActive: (rowId: string) => void;
  onRowRun: (row: PaletteRow) => void;
  /** Run a row's mod+Enter secondary action (row-level pointer affordance). */
  onRowRunSecondary: (row: PaletteRow) => void;
  onSearchChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onSearchKeyDown: (event: KeyboardEvent<HTMLInputElement>, scrollToIndex: ScrollToIndex) => void;
  placeholder: string;
  query: string;
  rows: PaletteRow[];
  scopeIsError: boolean;
  scopeIsFetching: boolean;
  scopeLabel: string | null;
  secondaryHint: string | undefined;
  setInputElement: (node: HTMLInputElement | null) => void;
  stage: PaletteStage | null;
  trimmedQuery: string;
}

export const useCommandPaletteController = ({
  entries,
  onClose,
  providers,
}: {
  entries: PaletteEntry[];
  onClose: () => void;
  providers: PaletteSearchProvider[];
}): CommandPaletteController => {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [activeRowId, setActiveRowId] = useState<string | null>(null);
  const [stage, setStage] = useState<PaletteStage | null>(null);
  const [scopeProviderKey, setScopeProviderKey] = useState<string | null>(null);
  const [recentIds] = useState(getRecentEntryIds);
  const inputRef = useRef<HTMLInputElement>(null);
  const debounceTimerRef = useRef<number | null>(null);

  useMountEffect(() => () => {
    if (debounceTimerRef.current !== null) {
      window.clearTimeout(debounceTimerRef.current);
    }
  });

  const setInputElement = useCallback((node: HTMLInputElement | null) => {
    inputRef.current = node;
    node?.focus();
  }, []);
  const focusInput = useCallback(() => {
    window.requestAnimationFrame(() => inputRef.current?.focus());
  }, []);
  const clearDebounce = useCallback(() => {
    if (debounceTimerRef.current !== null) {
      window.clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }
  }, []);

  const scopeProvider = scopeProviderKey
    ? (providers.find((provider) => provider.providerKey === scopeProviderKey) ?? null)
    : null;
  const isCommandsScope = !stage && !scopeProvider && query.startsWith('>');
  const localQuery = isCommandsScope ? query.slice(1) : query;
  const trimmedQuery = localQuery.trim();
  // Date tokens only mean anything when a provider can apply the range.
  // Everywhere else — commands scope, date-less scoped searches, hosts with
  // no capable providers (Launchpad) — they stay literal text, with no
  // affordances, so a filter is never silently dropped or falsely acknowledged.
  const datesEnabled =
    !stage &&
    !isCommandsScope &&
    (scopeProvider
      ? Boolean(scopeProvider.supportsCreatedAtRange)
      : providers.some((provider) => provider.supportsCreatedAtRange));
  // The raw query stays the source of truth; the parse is derived alongside it.
  const dateParse = useMemo(() => (datesEnabled ? parseDateTokens(localQuery) : null), [datesEnabled, localQuery]);
  const debouncedParse = useMemo(
    () => (datesEnabled ? parseDateTokens(debouncedQuery) : null),
    [datesEnabled, debouncedQuery]
  );
  const providerQuery = useMemo(
    () => ({ range: debouncedParse?.range, text: debouncedParse === null ? debouncedQuery : debouncedParse.text }),
    [debouncedParse, debouncedQuery]
  );
  // A valid date range is a complete query on its own — it bypasses the
  // minimum text length so `from:7d` alone searches. An active scope is a
  // committed search context: it searches at any length, including empty
  // (initial recent results) and one character.
  const shouldSearchProviders =
    (scopeProvider !== null ||
      providerQuery.text.length >= PROVIDER_MIN_QUERY_LENGTH ||
      providerQuery.range !== undefined) &&
    !stage &&
    !isCommandsScope;
  // Pure date query (range, no text): only range-capable providers run —
  // the rest would return broad, unfiltered results for empty text.
  const isPureDateQuery = providerQuery.range !== undefined && providerQuery.text.length === 0;
  const activeProviders = useMemo(() => {
    if (stage || isCommandsScope) {
      return NO_PROVIDERS;
    }

    const base = scopeProvider ? [scopeProvider] : providers;

    return isPureDateQuery ? base.filter((provider) => provider.supportsCreatedAtRange) : base;
  }, [isCommandsScope, isPureDateQuery, providers, scopeProvider, stage]);

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
      queryFn: () => Promise.resolve(provider.search(providerQuery)),
      queryKey: getPaletteProviderQueryKey(provider, providerQuery),
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

  const resetOverlayState = useCallback(() => {
    clearDebounce();
    setStage(null);
    setScopeProviderKey(null);
    setQuery('');
    setDebouncedQuery('');
    setActiveRowId(null);
  }, [clearDebounce]);
  const popOverlay = useCallback(() => {
    resetOverlayState();
    focusInput();
  }, [focusInput, resetOverlayState]);
  const onStageApplied = resetOverlayState;

  const previewRow = useCallback(
    (row: PaletteRow | undefined) => {
      if (!stage?.preview) {
        return;
      }

      if (row?.kind !== 'entry' || !row.entry.id.startsWith(STAGE_ENTRY_ID_PREFIX)) {
        stage.clearPreview?.();
        return;
      }

      stage.preview(row.entry.id.slice(STAGE_ENTRY_ID_PREFIX.length));
    },
    [stage]
  );

  // The single write path for the query: every mutation (typing, programmatic
  // completion) must go through here so the debounced provider query can never
  // go stale relative to the input. `immediate` skips the debounce for
  // discrete completions (suggestion rows) where results should refresh now.
  const replaceQuery = useCallback(
    (nextQuery: string, { immediate = false }: { immediate?: boolean } = {}) => {
      const nextLocalQuery = !stage && !scopeProvider && nextQuery.startsWith('>') ? nextQuery.slice(1) : nextQuery;
      const nextTrimmedQuery = nextLocalQuery.trim();

      setQuery(nextQuery);
      setActiveRowId(null);
      clearDebounce();

      if (stage) {
        const nextRows = searchPaletteRows(buildStageEntries(stage, onStageApplied), nextQuery, [], {
          showAllOnEmpty: true,
        });
        previewRow(nextRows.find((row) => row.kind !== 'label'));
      }

      if (immediate) {
        setDebouncedQuery(nextTrimmedQuery);
        return;
      }

      debounceTimerRef.current = window.setTimeout(() => {
        debounceTimerRef.current = null;
        setDebouncedQuery(nextTrimmedQuery);
      }, PROVIDER_DEBOUNCE_MS);
    },
    [clearDebounce, onStageApplied, previewRow, scopeProvider, stage]
  );

  // Completion rows for an in-progress trailing token (`from:` → Today /
  // Yesterday / …), rendered above all other sections. Selecting one rewrites
  // the token in place and refreshes provider results immediately.
  const dateSuggestionRows = useMemo<PaletteRow[]>(() => {
    if (!datesEnabled) {
      return [];
    }

    const trailing = matchTrailingDateToken(query);

    if (!trailing) {
      return [];
    }

    const partial = trailing.partialValue.toLowerCase();
    const options = DATE_TOKEN_SUGGESTIONS[trailing.key].filter((option) => option.value.startsWith(partial));
    const rows: PaletteRow[] = [{ id: 'label:date-suggestions', kind: 'label', label: 'Date' }];

    for (const option of options) {
      const id = `date-suggestion:${trailing.key}:${option.value}`;

      rows.push({
        entry: {
          group: 'Date',
          id,
          isPersistentRecent: false,
          keepOpen: true,
          run: () => replaceQuery(completeTrailingDateToken(query, option.value), { immediate: true }),
          subtitle: `${trailing.key}:${option.value}`,
          title: option.label,
        },
        id,
        kind: 'entry',
      });
    }

    rows.push({ id: 'label:date-format-hint', kind: 'label', label: DATE_FORMAT_HINT });

    return rows;
  }, [datesEnabled, query, replaceQuery]);

  const enterScope = useCallback(
    (providerKey: string, { resetQuery = false }: { resetQuery?: boolean } = {}) => {
      // A date-less scope cannot apply date tokens; hand it the token-stripped
      // text its scope row advertised, not the raw token text.
      const target = providers.find((provider) => provider.providerKey === providerKey);
      const keptQuery = !target?.supportsCreatedAtRange && dateParse !== null ? dateParse.text.trim() : trimmedQuery;

      clearDebounce();
      setScopeProviderKey(providerKey);
      setQuery(resetQuery ? '' : keptQuery);
      setDebouncedQuery(resetQuery ? '' : keptQuery);
      setActiveRowId(null);
      focusInput();
    },
    [clearDebounce, dateParse, focusInput, providers, trimmedQuery]
  );

  // One "Search images…"-style command per provider, merged into the local
  // entries so the empty-state launcher, `>` commands mode, fuzzy matching,
  // and recents all treat them like any other command. Running one enters the
  // scope with a fresh query — the text that found the command is not a
  // search term for its results.
  const scopeCommandEntries = useMemo<PaletteEntry[]>(
    () =>
      providers.map((provider) => ({
        group: SEARCH_SCOPE_GROUP,
        id: getPaletteContributionKey('scope-command', provider.providerKey),
        isPersistentRecent: true,
        keepOpen: true,
        keywords: `search find browse ${provider.label.toLowerCase()}`,
        run: () => enterScope(provider.providerKey, { resetQuery: true }),
        showInEmptyState: true,
        title: `Search ${provider.label.toLowerCase()}…`,
      })),
    [enterScope, providers]
  );
  const allEntries = useMemo(() => [...entries, ...scopeCommandEntries], [entries, scopeCommandEntries]);

  const rows = useMemo<PaletteRow[]>(() => {
    if (stage) {
      return searchPaletteRows(buildStageEntries(stage, onStageApplied), query, [], { showAllOnEmpty: true });
    }

    if (scopeProvider) {
      return buildProviderSectionRows(providerSections, null);
    }

    const matchText = dateParse === null ? localQuery : dateParse.text;
    const hasLiveRange = dateParse !== null && dateParse.range !== undefined;
    const isLivePureDate = hasLiveRange && matchText.trim().length === 0;
    // A pure date query must not fall back to the recents launcher; its rows
    // are provider results (and their scopes) only.
    const localRows = isLivePureDate
      ? []
      : searchPaletteRows(allEntries, matchText, recentIds, {
          commandsOnly: isCommandsScope,
          showAllOnEmpty: isCommandsScope && trimmedQuery.length === 0,
        });

    if (isCommandsScope) {
      return localRows;
    }

    if (trimmedQuery.length === 0) {
      // Quiet trailing tip in the launcher — the only place the query syntax
      // is advertised before it is typed.
      const syntaxHint = providers.some((provider) => provider.supportsCreatedAtRange)
        ? 'Type > for commands · from:/date: to filter by date'
        : 'Type > for commands';

      return [...localRows, { id: 'label:syntax-hint', kind: 'label', label: syntaxHint }];
    }

    const scopeProviders = isLivePureDate ? providers.filter((provider) => provider.supportsCreatedAtRange) : providers;

    return [
      ...dateSuggestionRows,
      ...localRows,
      ...(matchText.trim().length >= PROVIDER_MIN_QUERY_LENGTH || hasLiveRange
        ? buildProviderSectionRows(providerSections)
        : []),
      ...buildScopeRows(scopeProviders, isLivePureDate ? '' : matchText.trim()),
    ];
  }, [
    allEntries,
    dateParse,
    dateSuggestionRows,
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
  const navigableRows = useMemo(
    () => rows.flatMap((row, index) => (row.kind === 'label' ? [] : [{ index, row }])),
    [rows]
  );
  const effectiveActive = useMemo(() => resolveActivePaletteRow(rows, activeRowId), [activeRowId, rows]);
  const effectiveActiveRowId = effectiveActive?.row.id ?? null;
  const activeRow = effectiveActive?.row;

  const runRow = useCallback(
    (row: PaletteRow) => {
      if (row.kind === 'label') {
        return;
      }

      if (row.kind === 'scope') {
        enterScope(row.providerKey);
        return;
      }

      if (row.entry.isPersistentRecent) {
        recordRecentEntry(row.entry);
      }

      if (row.entry.stage) {
        const nextStage = row.entry.stage;
        const currentOption = nextStage.options.find((option) => option.isCurrent);

        clearDebounce();
        setStage(nextStage);
        setQuery('');
        setDebouncedQuery('');
        setActiveRowId(currentOption ? `${STAGE_ENTRY_ID_PREFIX}${currentOption.id}` : null);
        focusInput();
        return;
      }

      if (row.entry.keepOpen) {
        void row.entry.run();
        focusInput();
        return;
      }

      onClose();
      void row.entry.run();
    },
    [clearDebounce, enterScope, focusInput, onClose]
  );

  const runSecondary = useCallback(
    (entry: PaletteEntry) => {
      if (!entry.secondary) {
        return;
      }

      if (entry.isPersistentRecent) {
        recordRecentEntry(entry);
      }
      onClose();
      void entry.secondary.run();
    },
    [onClose]
  );

  const onRowRunSecondary = useCallback(
    (row: PaletteRow) => {
      if (row.kind === 'entry' && row.entry.secondary) {
        runSecondary(row.entry);
      }
    },
    [runSecondary]
  );

  const moveActive = useCallback(
    (offset: number, scrollToIndex: ScrollToIndex) => {
      if (navigableRows.length === 0) {
        return;
      }

      const position = effectiveActive
        ? navigableRows.findIndex((candidate) => candidate.row.id === effectiveActive.row.id)
        : -1;
      const nextPosition =
        Math.abs(offset) === 1
          ? (position + offset + navigableRows.length) % navigableRows.length
          : Math.min(Math.max(position + offset, 0), navigableRows.length - 1);
      const next = navigableRows[nextPosition];

      if (!next) {
        return;
      }

      setActiveRowId(next.row.id);
      previewRow(next.row);
      scrollToIndex(next.index);
    },
    [effectiveActive, navigableRows, previewRow]
  );

  const onEscape = useCallback(() => {
    if (stage || scopeProviderKey) {
      popOverlay();
      return;
    }

    onClose();
  }, [onClose, popOverlay, scopeProviderKey, stage]);

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
    (event: KeyboardEvent<HTMLInputElement>, scrollToIndex: ScrollToIndex) => {
      if (event.key === 'ArrowDown') {
        event.preventDefault();
        moveActive(1, scrollToIndex);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        moveActive(-1, scrollToIndex);
      } else if (event.key === 'PageDown') {
        event.preventDefault();
        moveActive(PAGE_JUMP_ROWS, scrollToIndex);
      } else if (event.key === 'PageUp') {
        event.preventDefault();
        moveActive(-PAGE_JUMP_ROWS, scrollToIndex);
      } else if (event.key === 'Tab' && activeRow?.kind === 'scope') {
        event.preventDefault();
        enterScope(activeRow.providerKey);
      } else if (event.key === 'Backspace' && query.length === 0 && (stage || scopeProviderKey)) {
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
    [activeRow, enterScope, moveActive, popOverlay, query.length, runRow, runSecondary, scopeProviderKey, stage]
  );

  const onSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => replaceQuery(event.currentTarget.value),
    [replaceQuery]
  );

  const onRowActive = useCallback(
    (rowId: string) => {
      const row = rows.find((candidate) => candidate.id === rowId);
      setActiveRowId(rowId);
      previewRow(row);
    },
    [previewRow, rows]
  );

  const onRetry = useCallback(() => {
    void scopedQueryResult?.refetch();
    focusInput();
  }, [focusInput, scopedQueryResult]);

  // Quiet acknowledgment of the applied range, and inline feedback for values
  // that cannot parse. A trailing token that could still become valid
  // (`from:`, `from:2026-`) is normal typing, not an error. Feedback beats
  // acknowledgment when both apply.
  const dateInvalidHint = useMemo(() => {
    if (dateParse === null) {
      return null;
    }

    const trailing = matchTrailingDateToken(localQuery);
    const invalid = dateParse.invalidTokens.find(
      (token) =>
        !(
          trailing &&
          token.key === trailing.key &&
          token.raw === trailing.partialValue &&
          isPossibleDatePrefix(token.raw)
        )
    );

    return invalid ? `Invalid date: “${invalid.raw}”` : null;
  }, [dateParse, localQuery]);
  const dateSummary = dateParse?.range && dateInvalidHint === null ? formatDateRangeLabel(dateParse.range) : null;

  return {
    dateInvalidHint,
    dateSummary,
    activeRow,
    activeRowId: effectiveActiveRowId,
    chipLabel: stage ? stage.title : (scopeProvider?.label ?? null),
    hasScopeRows: rows.some((row) => row.kind === 'scope'),
    isOverlayOpen: Boolean(stage || scopeProviderKey),
    onContentKeyDown,
    onPopOverlay: popOverlay,
    onRetry,
    onRowActive,
    onRowRun: runRow,
    onRowRunSecondary,
    onSearchChange,
    onSearchKeyDown,
    placeholder: stage
      ? 'Pick a value…'
      : scopeProvider
        ? `Search ${scopeProvider.label.toLowerCase()}…`
        : 'Search commands, settings, and more…',
    query,
    rows,
    scopeIsError: Boolean(scopeProvider && scopedQueryResult?.isError),
    scopeIsFetching: Boolean(scopeProvider && scopedQueryResult?.isFetching),
    scopeLabel: scopeProvider?.label.toLowerCase() ?? null,
    secondaryHint: activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined,
    setInputElement,
    stage,
    trimmedQuery,
  };
};
