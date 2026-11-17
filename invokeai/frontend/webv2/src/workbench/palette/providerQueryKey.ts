import type { PaletteProviderQuery, PaletteSearchProvider } from './entries';

/** Keyed on stripped text plus the resolved date range, so relative tokens cannot hit a stale entry. */
export const getPaletteProviderQueryKey = (provider: PaletteSearchProvider, query: PaletteProviderQuery) =>
  [
    'command-palette',
    provider.providerKey,
    provider.contextKey,
    query.text,
    query.range?.from ?? '',
    query.range?.to ?? '',
  ] as const;
