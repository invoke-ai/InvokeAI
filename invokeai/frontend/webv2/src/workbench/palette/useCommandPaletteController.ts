import type { DateTokenKey } from '@platform/search/dateTokens';
import type { ChangeEvent, KeyboardEvent } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import {
  completeTrailingDateToken,
  describeDateRange,
  findInvalidDateToken,
  formatIsoDate,
  matchTrailingDateToken,
  parseDateTokens,
} from '@platform/search/dateTokens';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, PaletteProviderQuery, PaletteRow, PaletteSearchProvider, PaletteStage } from './entries';

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
import { PaletteDebouncer } from './paletteDebouncer';
import {
  changePaletteSelection,
  commitPaletteQuery,
  createInitialPaletteState,
  enterPaletteScope,
  enterPaletteStage,
  replacePaletteQuery,
  returnPaletteToRoot,
} from './paletteState';
import { getRecentEntryIds, recordRecentEntry } from './recents';
import { usePaletteProviderSections } from './usePaletteProviderSections';

const PAGE_JUMP_ROWS = 8;
const PROVIDER_DEBOUNCE_MS = 200;
const NO_PROVIDERS: PaletteSearchProvider[] = [];
export const COMMAND_PALETTE_INPUT_ID = 'command-palette-query';

/**
 * Completions offered while a date token is being typed. `Nd` resolves to a
 * single date N days ago, so the labels are key-specific: `from:7d` is a
 * starting point ("Past week"), while `date:7d` is one day ("A week ago").
 */
const DATE_TOKEN_SUGGESTIONS: Record<DateTokenKey, ReadonlyArray<{ labelKey: string; value: string }>> = {
  date: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
    { labelKey: 'commandPalette.date.aWeekAgo', value: '7d' },
  ],
  from: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
    { labelKey: 'commandPalette.date.pastWeek', value: '7d' },
  ],
  to: [
    { labelKey: 'commandPalette.date.today', value: 'today' },
    { labelKey: 'commandPalette.date.yesterday', value: 'yesterday' },
  ],
};

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
  onSearchChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onSearchKeyDown: (event: KeyboardEvent<HTMLInputElement>, scrollToIndex: ScrollToIndex) => void;
  placeholder: string;
  query: string;
  rows: PaletteRow[];
  scopeIsError: boolean;
  scopeIsFetching: boolean;
  scopeLabel: string | null;
  secondaryHint: string | undefined;
  stage: PaletteStage | null;
  trimmedQuery: string;
  statusMessage: string;
  isBusy: boolean;
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
  const { i18n, t } = useTranslation();
  const [paletteState, setPaletteState] = useState(createInitialPaletteState);
  const [recentIds] = useState(getRecentEntryIds);
  const [debouncer] = useState(
    () =>
      new PaletteDebouncer(PROVIDER_DEBOUNCE_MS, (value) => {
        setPaletteState((current) => commitPaletteQuery(current, value));
      })
  );

  useMountEffect(() => debouncer.cancel);

  const focusInput = useCallback(() => {
    window.requestAnimationFrame(() => document.getElementById(COMMAND_PALETTE_INPUT_ID)?.focus());
  }, []);
  const clearDebounce = debouncer.cancel;

  const requestedScopeProviderKey = paletteState.mode.kind === 'scoped' ? paletteState.mode.providerKey : null;
  const scopeProvider = requestedScopeProviderKey
    ? (providers.find((provider) => provider.providerKey === requestedScopeProviderKey) ?? null)
    : null;
  const resolvedMode =
    paletteState.mode.kind === 'scoped' && scopeProvider === null ? { kind: 'root' as const } : paletteState.mode;
  const stage = resolvedMode.kind === 'stage' ? resolvedMode.stage : null;
  const scopeProviderKey = resolvedMode.kind === 'scoped' ? resolvedMode.providerKey : null;
  const { activeRowId, debouncedQuery, query } = paletteState;
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
  const providerQuery = useMemo<PaletteProviderQuery>(
    () => ({
      range: debouncedParse?.range,
      text: (debouncedParse === null ? debouncedQuery : debouncedParse.text).trim(),
    }),
    [debouncedParse, debouncedQuery]
  );
  const liveProviderQuery = useMemo<PaletteProviderQuery>(
    () => ({ range: dateParse?.range, text: (dateParse === null ? localQuery : dateParse.text).trim() }),
    [dateParse, localQuery]
  );
  const isWaitingForDebounce =
    providerQuery.text !== liveProviderQuery.text ||
    providerQuery.range?.from !== liveProviderQuery.range?.from ||
    providerQuery.range?.to !== liveProviderQuery.range?.to;
  // A valid date range is a complete query on its own — it bypasses the
  // minimum text length so `from:7d` alone searches. An active scope is a
  // committed search context: it searches at any length, including empty
  // (initial recent results) and one character.
  const shouldSearchProviders =
    (scopeProvider !== null ||
      liveProviderQuery.text.length >= PROVIDER_MIN_QUERY_LENGTH ||
      liveProviderQuery.range !== undefined) &&
    !stage &&
    !isCommandsScope;
  // Pure date query (range, no text): only range-capable providers run —
  // the rest would return broad, unfiltered results for empty text.
  const isPureDateQuery = liveProviderQuery.range !== undefined && liveProviderQuery.text.length === 0;
  const activeProviders = useMemo(() => {
    if (stage || isCommandsScope) {
      return NO_PROVIDERS;
    }

    const base = scopeProvider ? [scopeProvider] : providers;

    return isPureDateQuery ? base.filter((provider) => provider.supportsCreatedAtRange) : base;
  }, [isCommandsScope, isPureDateQuery, providers, scopeProvider, stage]);

  const { results: providerResults, sections: providerSections } = usePaletteProviderSections({
    enabled: shouldSearchProviders && !isWaitingForDebounce,
    isWaitingForDebounce: shouldSearchProviders && isWaitingForDebounce,
    providerQuery,
    providers: activeProviders,
  });
  const scopedQueryResult = scopeProvider ? providerResults[0] : undefined;

  const resetOverlayState = useCallback(() => {
    clearDebounce();
    setPaletteState(returnPaletteToRoot());
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

      setPaletteState((current) => replacePaletteQuery(current, nextQuery));
      clearDebounce();

      if (stage) {
        const nextRows = searchPaletteRows(buildStageEntries(stage, onStageApplied), nextQuery, [], {
          showAllOnEmpty: true,
        });
        previewRow(nextRows.find((row) => row.kind !== 'label'));
      }

      if (immediate) {
        debouncer.commit(nextTrimmedQuery);
        return;
      }

      debouncer.schedule(nextTrimmedQuery);
    },
    [clearDebounce, debouncer, onStageApplied, previewRow, scopeProvider, stage]
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
    const rows: PaletteRow[] = [{ id: 'label:date-suggestions', kind: 'label', label: t('commandPalette.date.date') }];

    for (const option of options) {
      const id = `date-suggestion:${trailing.key}:${option.value}`;

      rows.push({
        entry: {
          group: 'Date',
          groupLabel: t('commandPalette.groups.date'),
          id,
          isPersistentRecent: false,
          keepOpen: true,
          run: () => replaceQuery(completeTrailingDateToken(query, option.value), { immediate: true }),
          subtitle: `${trailing.key}:${option.value}`,
          title: t(option.labelKey),
        },
        id,
        kind: 'entry',
      });
    }

    rows.push({ id: 'label:date-format-hint', kind: 'label', label: t('commandPalette.date.formatHint') });

    return rows;
  }, [datesEnabled, query, replaceQuery, t]);

  const enterScope = useCallback(
    (providerKey: string, { resetQuery = false }: { resetQuery?: boolean } = {}) => {
      // A date-less scope cannot apply date tokens; hand it the token-stripped
      // text its scope row advertised, not the raw token text.
      const target = providers.find((provider) => provider.providerKey === providerKey);
      const keptQuery = !target?.supportsCreatedAtRange && dateParse !== null ? dateParse.text.trim() : trimmedQuery;

      clearDebounce();
      setPaletteState(enterPaletteScope(providerKey, resetQuery ? '' : keptQuery));
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
        groupLabel: t('commandPalette.search.in'),
        id: getPaletteContributionKey('scope-command', provider.providerKey),
        isPersistentRecent: true,
        keepOpen: true,
        keywords: `search find browse ${provider.label.toLowerCase()}`,
        run: () => enterScope(provider.providerKey, { resetQuery: true }),
        showInEmptyState: true,
        title: t('commandPalette.search.provider', { label: provider.label.toLowerCase() }),
      })),
    [enterScope, providers, t]
  );
  const allEntries = useMemo(() => [...entries, ...scopeCommandEntries], [entries, scopeCommandEntries]);

  const rows = useMemo<PaletteRow[]>(() => {
    if (stage) {
      return searchPaletteRows(buildStageEntries(stage, onStageApplied, t), query, [], { showAllOnEmpty: true });
    }

    if (scopeProvider) {
      return buildProviderSectionRows(providerSections, null, t);
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
          recentLabel: t('commandPalette.groups.recent'),
          showAllOnEmpty: isCommandsScope && trimmedQuery.length === 0,
        });

    if (isCommandsScope) {
      return localRows;
    }

    if (trimmedQuery.length === 0) {
      // Quiet trailing tip in the launcher — the only place the query syntax
      // is advertised before it is typed.
      const syntaxHint = providers.some((provider) => provider.supportsCreatedAtRange)
        ? t('commandPalette.syntax.commandsAndDates')
        : t('commandPalette.syntax.commands');

      return [...localRows, { id: 'label:syntax-hint', kind: 'label', label: syntaxHint }];
    }

    const scopeProviders = isLivePureDate ? providers.filter((provider) => provider.supportsCreatedAtRange) : providers;

    return [
      ...dateSuggestionRows,
      ...localRows,
      ...(matchText.trim().length >= PROVIDER_MIN_QUERY_LENGTH || hasLiveRange
        ? buildProviderSectionRows(providerSections, undefined, t)
        : []),
      ...buildScopeRows(scopeProviders, isLivePureDate ? '' : matchText.trim(), t),
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
    t,
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

      if (row.kind === 'provider-error') {
        row.retry();
        focusInput();
        return;
      }

      if (row.entry.isPersistentRecent) {
        recordRecentEntry(row.entry);
      }

      if (row.entry.stage) {
        const nextStage = row.entry.stage;
        const currentOption = nextStage.options.find((option) => option.isCurrent);

        clearDebounce();
        setPaletteState(
          enterPaletteStage(nextStage, currentOption ? `${STAGE_ENTRY_ID_PREFIX}${currentOption.id}` : null)
        );
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

      setPaletteState((current) => changePaletteSelection(current, next.row.id));
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
      setPaletteState((current) => changePaletteSelection(current, rowId));
      previewRow(row);
    },
    [previewRow, rows]
  );

  const onRetry = useCallback(() => {
    scopedQueryResult?.retry();
    focusInput();
  }, [focusInput, scopedQueryResult]);

  // Quiet acknowledgment of the applied range, and inline feedback for values
  // that cannot parse. Feedback beats acknowledgment when both apply.
  const dateInvalidHint = useMemo(() => {
    const invalid = dateParse === null ? undefined : findInvalidDateToken(localQuery, dateParse);

    return invalid ? t('commandPalette.states.invalidDate', { value: invalid.raw }) : null;
  }, [dateParse, localQuery, t]);
  const dateShape = dateParse?.range && dateInvalidHint === null ? describeDateRange(dateParse.range) : null;
  const dateSummary =
    dateShape === null
      ? null
      : dateShape.kind === 'day'
        ? t('commandPalette.date.day', { date: formatIsoDate(dateShape.date, i18n.resolvedLanguage) })
        : dateShape.kind === 'range'
          ? t('commandPalette.date.range', {
              from: formatIsoDate(dateShape.from, i18n.resolvedLanguage),
              to: formatIsoDate(dateShape.to, i18n.resolvedLanguage),
            })
          : dateShape.kind === 'from'
            ? t('commandPalette.date.from', { date: formatIsoDate(dateShape.date, i18n.resolvedLanguage) })
            : t('commandPalette.date.through', { date: formatIsoDate(dateShape.date, i18n.resolvedLanguage) });

  const isBusy = providerSections.some((section) => section.isFetching || section.isWaitingForDebounce);
  const failedProviderCount = providerSections.filter((section) => section.isError).length;
  const providerResultCount = providerSections.reduce((count, section) => count + section.entries.length, 0);
  const statusMessage = isBusy
    ? t('commandPalette.states.searching')
    : failedProviderCount > 0
      ? t('commandPalette.states.partialFailure', { count: providerResultCount, failed: failedProviderCount })
      : t('commandPalette.states.resultCount', { count: providerResultCount });

  return {
    dateInvalidHint,
    dateSummary,
    activeRow,
    activeRowId: effectiveActiveRowId,
    chipLabel: stage ? stage.title : (scopeProvider?.label ?? null),
    hasScopeRows: rows.some((row) => row.kind === 'scope'),
    isOverlayOpen: resolvedMode.kind !== 'root',
    isBusy,
    onContentKeyDown,
    onPopOverlay: popOverlay,
    onRetry,
    onRowActive,
    onRowRun: runRow,
    onSearchChange,
    onSearchKeyDown,
    placeholder: stage
      ? t('commandPalette.placeholders.stage')
      : scopeProvider
        ? t('commandPalette.placeholders.scope', { label: scopeProvider.label.toLowerCase() })
        : t('commandPalette.placeholders.root'),
    query,
    rows,
    scopeIsError: Boolean(scopeProvider && scopedQueryResult?.isError),
    scopeIsFetching: Boolean(scopeProvider && (scopedQueryResult?.isFetching || isWaitingForDebounce)),
    scopeLabel: scopeProvider?.label.toLowerCase() ?? null,
    secondaryHint: activeRow?.kind === 'entry' ? activeRow.entry.secondary?.label : undefined,
    stage,
    statusMessage,
    trimmedQuery,
  };
};
