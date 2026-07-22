import type { ChangeEvent, KeyboardEvent } from 'react';

import { useMountEffect } from '@platform/react/useMountEffect';
import { describeDateRange, findInvalidDateToken, formatIsoDate } from '@platform/search/dateTokens';
import { useCallback, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import type { PaletteEntry, PaletteRow, PaletteSearchProvider, PaletteStage } from './entries';
import type { ScrollToIndex } from './usePaletteNavigation';

import { buildStageEntries, searchPaletteRows, STAGE_ENTRY_ID_PREFIX } from './entries';
import { PaletteDebouncer } from './paletteDebouncer';
import { derivePaletteQueryModel } from './paletteQueryModel';
import { buildCommandPaletteRows } from './paletteRowModel';
import {
  commitPaletteQuery,
  createInitialPaletteState,
  enterPaletteScope,
  replacePaletteQuery,
  returnPaletteToRoot,
} from './paletteState';
import { getRecentEntryIds } from './recents';
import { usePaletteNavigation } from './usePaletteNavigation';
import { usePaletteProviderSections } from './usePaletteProviderSections';

const PROVIDER_DEBOUNCE_MS = 200;
export const COMMAND_PALETTE_INPUT_ID = 'command-palette-query';

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
  const { activeRowId, debouncedQuery, mode, query } = paletteState;
  const queryModel = useMemo(
    () => derivePaletteQueryModel({ providers, state: { debouncedQuery, mode, query } }),
    [debouncedQuery, mode, providers, query]
  );
  const {
    activeProviders,
    dateParse,
    isWaitingForDebounce,
    localQuery,
    providerQuery,
    resolvedMode,
    scopeProvider,
    scopeProviderKey,
    shouldSearchProviders,
    stage,
    trimmedQuery,
  } = queryModel;
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

  const replaceQuery = useCallback(
    (nextQuery: string, { immediate = false }: { immediate?: boolean } = {}) => {
      const nextLocalQuery = !stage && !scopeProvider && nextQuery.startsWith('>') ? nextQuery.slice(1) : nextQuery;
      const nextTrimmedQuery = nextLocalQuery.trim();

      setPaletteState((current) => replacePaletteQuery(current, nextQuery));
      clearDebounce();

      if (stage) {
        const nextRows = searchStageRows(stage, onStageApplied, nextQuery);
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

  const enterScope = useCallback(
    (providerKey: string, { resetQuery = false }: { resetQuery?: boolean } = {}) => {
      const target = providers.find((provider) => provider.providerKey === providerKey);
      const keptQuery = !target?.supportsCreatedAtRange && dateParse !== null ? dateParse.text.trim() : trimmedQuery;

      clearDebounce();
      setPaletteState(enterPaletteScope(providerKey, resetQuery ? '' : keptQuery));
      focusInput();
    },
    [clearDebounce, dateParse, focusInput, providers, trimmedQuery]
  );

  const rows = useMemo(
    () =>
      buildCommandPaletteRows({
        enterScope,
        entries,
        onCompleteDateSuggestion: (nextQuery) => replaceQuery(nextQuery, { immediate: true }),
        onStageApplied,
        providers,
        providerSections,
        queryModel,
        recentIds,
        t,
      }),
    [enterScope, entries, onStageApplied, providerSections, providers, queryModel, recentIds, replaceQuery, t]
  );

  const navigation = usePaletteNavigation({
    activeRowId,
    clearDebounce,
    enterScope,
    focusInput,
    onClose,
    popOverlay,
    previewRow,
    query,
    rows,
    scopeProviderKey,
    setPaletteState,
    stage,
  });

  const onSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => replaceQuery(event.currentTarget.value),
    [replaceQuery]
  );

  const onRetry = useCallback(() => {
    scopedQueryResult?.retry();
    focusInput();
  }, [focusInput, scopedQueryResult]);

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
    activeRow: navigation.activeRow,
    activeRowId: navigation.activeRowId,
    chipLabel: stage ? stage.title : (scopeProvider?.label ?? null),
    dateInvalidHint,
    dateSummary,
    hasScopeRows: rows.some((row) => row.kind === 'scope'),
    isBusy,
    isOverlayOpen: resolvedMode.kind !== 'root',
    onContentKeyDown: navigation.onContentKeyDown,
    onPopOverlay: popOverlay,
    onRetry,
    onRowActive: navigation.onRowActive,
    onRowRun: navigation.onRowRun,
    onSearchChange,
    onSearchKeyDown: navigation.onSearchKeyDown,
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
    secondaryHint: navigation.secondaryHint,
    stage,
    statusMessage,
    trimmedQuery,
  };
};

const searchStageRows = (stage: PaletteStage, onStageApplied: () => void, query: string): PaletteRow[] =>
  // Kept local because stage preview needs the next synchronous row set while typing.
  // The main rendered row path lives in paletteRowModel.
  searchPaletteRows(buildStageEntries(stage, onStageApplied), query, [], { showAllOnEmpty: true });
