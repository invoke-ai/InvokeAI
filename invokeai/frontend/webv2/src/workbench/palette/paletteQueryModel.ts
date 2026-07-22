import type { DateTokenParse } from '@platform/search/dateTokens';

import { parseDateTokens } from '@platform/search/dateTokens';

import type { PaletteProviderQuery, PaletteSearchProvider, PaletteStage } from './entries';
import type { PaletteMode, PaletteState } from './paletteState';

import { PROVIDER_MIN_QUERY_LENGTH } from './entries';

const NO_PROVIDERS: PaletteSearchProvider[] = [];

export interface PaletteQueryModel {
  activeProviders: PaletteSearchProvider[];
  dateParse: DateTokenParse | null;
  datesEnabled: boolean;
  isCommandsScope: boolean;
  isPureDateQuery: boolean;
  isWaitingForDebounce: boolean;
  liveProviderQuery: PaletteProviderQuery;
  localQuery: string;
  providerQuery: PaletteProviderQuery;
  query: string;
  resolvedMode: PaletteMode;
  scopeProvider: PaletteSearchProvider | null;
  scopeProviderKey: string | null;
  shouldSearchProviders: boolean;
  stage: PaletteStage | null;
  trimmedQuery: string;
}

export const derivePaletteQueryModel = ({
  providers,
  state,
}: {
  providers: PaletteSearchProvider[];
  state: Pick<PaletteState, 'debouncedQuery' | 'mode' | 'query'>;
}): PaletteQueryModel => {
  const requestedScopeProviderKey = state.mode.kind === 'scoped' ? state.mode.providerKey : null;
  const scopeProvider = requestedScopeProviderKey
    ? (providers.find((provider) => provider.providerKey === requestedScopeProviderKey) ?? null)
    : null;
  const resolvedMode =
    state.mode.kind === 'scoped' && scopeProvider === null ? ({ kind: 'root' } as const) : state.mode;
  const stage = resolvedMode.kind === 'stage' ? resolvedMode.stage : null;
  const scopeProviderKey = resolvedMode.kind === 'scoped' ? resolvedMode.providerKey : null;
  const isCommandsScope = !stage && !scopeProvider && state.query.startsWith('>');
  const localQuery = isCommandsScope ? state.query.slice(1) : state.query;
  const trimmedQuery = localQuery.trim();
  const datesEnabled =
    !stage &&
    !isCommandsScope &&
    (scopeProvider
      ? Boolean(scopeProvider.supportsCreatedAtRange)
      : providers.some((provider) => provider.supportsCreatedAtRange));
  const dateParse = datesEnabled ? parseDateTokens(localQuery) : null;
  const debouncedParse = datesEnabled ? parseDateTokens(state.debouncedQuery) : null;
  const providerQuery: PaletteProviderQuery = {
    range: debouncedParse?.range,
    text: (debouncedParse === null ? state.debouncedQuery : debouncedParse.text).trim(),
  };
  const liveProviderQuery: PaletteProviderQuery = {
    range: dateParse?.range,
    text: (dateParse === null ? localQuery : dateParse.text).trim(),
  };
  const isWaitingForDebounce =
    providerQuery.text !== liveProviderQuery.text ||
    providerQuery.range?.from !== liveProviderQuery.range?.from ||
    providerQuery.range?.to !== liveProviderQuery.range?.to;
  const shouldSearchProviders =
    (scopeProvider !== null ||
      liveProviderQuery.text.length >= PROVIDER_MIN_QUERY_LENGTH ||
      liveProviderQuery.range !== undefined) &&
    !stage &&
    !isCommandsScope;
  const isPureDateQuery = liveProviderQuery.range !== undefined && liveProviderQuery.text.length === 0;
  const baseProviders = scopeProvider ? [scopeProvider] : providers;
  const activeProviders =
    stage || isCommandsScope
      ? NO_PROVIDERS
      : isPureDateQuery
        ? baseProviders.filter((provider) => provider.supportsCreatedAtRange)
        : baseProviders;

  return {
    activeProviders,
    dateParse,
    datesEnabled,
    isCommandsScope,
    isPureDateQuery,
    isWaitingForDebounce,
    liveProviderQuery,
    localQuery,
    providerQuery,
    query: state.query,
    resolvedMode,
    scopeProvider,
    scopeProviderKey,
    shouldSearchProviders,
    stage,
    trimmedQuery,
  };
};
