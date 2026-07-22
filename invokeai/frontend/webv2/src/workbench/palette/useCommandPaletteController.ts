/* eslint-disable react/react-compiler */
import type { ChangeEvent, KeyboardEvent } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { parseDateTokens } from '@platform/search/dateTokens';
import { useQueries } from '@tanstack/react-query';
import { useCallback, useMemo, useRef, useState } from 'react';

import type { PaletteEntry, PaletteRow, PaletteSearchProvider, PaletteStage, ProviderResultSection } from './entries';

import {
  buildProviderSectionRows,
  buildScopeRows,
  buildStageEntries,
  PROVIDER_MIN_QUERY_LENGTH,
  resolveActivePaletteRow,
  searchPaletteRows,
  STAGE_ENTRY_ID_PREFIX,
} from './entries';
import { getPaletteProviderQueryKey } from './providerQueryKey';
import { getRecentEntryIds, recordRecentEntry } from './recents';

const PAGE_JUMP_ROWS = 8;
const PROVIDER_DEBOUNCE_MS = 200;
const NO_PROVIDERS: PaletteSearchProvider[] = [];

type ScrollToIndex = (index: number) => void;

export interface CommandPaletteController {
  activeRow: PaletteRow | undefined;
  activeRowId: string | null;
  chipLabel: string | null;
  hasScopeRows: boolean;
  isOverlayOpen: boolean;
  onContentKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  onRetry: () => void;
  onRowActive: (rowId: string) => void;
  onRowRun: (row: PaletteRow) => void;
  onSearchChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onSearchKeyDown: (event: KeyboardEvent<HTMLInputElement>, scrollToIndex: ScrollToIndex) => void;
  placeholder: string;
  query: string;
  rows: PaletteRow[];
  scopeErrorLabel: string | null;
  scopeIsError: boolean;
  scopeIsFetching: boolean;
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
  // In commands scope date tokens are literal text; everywhere else the raw
  // query stays the source of truth and the parse is derived alongside it.
  const dateParse = useMemo(
    () => (isCommandsScope ? null : parseDateTokens(localQuery)),
    [isCommandsScope, localQuery]
  );
  const debouncedParse = useMemo(() => parseDateTokens(debouncedQuery), [debouncedQuery]);
  const providerQuery = useMemo(() => ({ range: debouncedParse.range, text: debouncedParse.text }), [debouncedParse]);
  // A valid date range is a complete query on its own — it bypasses the
  // minimum text length so `from:7d` alone searches.
  const shouldSearchProviders =
    (debouncedParse.text.length >= PROVIDER_MIN_QUERY_LENGTH || debouncedParse.range !== undefined) &&
    !stage &&
    !isCommandsScope;
  // Pure date query (range, no text): only range-capable providers run —
  // the rest would return broad, unfiltered results for empty text.
  const isPureDateQuery = debouncedParse.range !== undefined && debouncedParse.text.length === 0;
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
      : searchPaletteRows(entries, matchText, recentIds, {
          commandsOnly: isCommandsScope,
          showAllOnEmpty: isCommandsScope && trimmedQuery.length === 0,
        });

    if (trimmedQuery.length === 0 || isCommandsScope) {
      return localRows;
    }

    const scopeProviders = isLivePureDate ? providers.filter((provider) => provider.supportsCreatedAtRange) : providers;

    return [
      ...localRows,
      ...(matchText.trim().length >= PROVIDER_MIN_QUERY_LENGTH || hasLiveRange
        ? buildProviderSectionRows(providerSections)
        : []),
      ...buildScopeRows(scopeProviders, isLivePureDate ? '' : matchText.trim()),
    ];
  }, [
    dateParse,
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
  const navigableRows = useMemo(
    () => rows.flatMap((row, index) => (row.kind === 'label' ? [] : [{ index, row }])),
    [rows]
  );
  const effectiveActive = useMemo(() => resolveActivePaletteRow(rows, activeRowId), [activeRowId, rows]);
  const effectiveActiveRowId = effectiveActive?.row.id ?? null;
  const activeRow = effectiveActive?.row;

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

  const enterScope = useCallback(
    (providerKey: string) => {
      clearDebounce();
      setScopeProviderKey(providerKey);
      setDebouncedQuery(trimmedQuery);
      setActiveRowId(null);
      focusInput();
    },
    [clearDebounce, focusInput, trimmedQuery]
  );

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

  return {
    activeRow,
    activeRowId: effectiveActiveRowId,
    chipLabel: stage ? stage.title : (scopeProvider?.label ?? null),
    hasScopeRows: rows.some((row) => row.kind === 'scope'),
    isOverlayOpen: Boolean(stage || scopeProviderKey),
    onContentKeyDown,
    onRetry,
    onRowActive,
    onRowRun: runRow,
    onSearchChange,
    onSearchKeyDown,
    placeholder: stage
      ? 'Pick a value…'
      : scopeProvider
        ? `Search ${scopeProvider.label.toLowerCase()}…`
        : 'Search commands, settings, and more…',
    query,
    rows,
    scopeErrorLabel: scopeProvider?.label.toLowerCase() ?? null,
    scopeIsError: Boolean(scopeProvider && scopedQueryResult?.isError),
    scopeIsFetching: Boolean(scopeProvider && scopedQueryResult?.isFetching),
    secondaryHint: activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined,
    setInputElement,
    stage,
    trimmedQuery,
  };
};
