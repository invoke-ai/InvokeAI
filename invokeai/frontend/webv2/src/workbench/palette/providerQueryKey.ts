import type { PaletteProviderQuery, PaletteSearchProvider } from './entries';

/**
 * Keyed on the semantic query — stripped text plus the *resolved* date range —
 * so relative tokens (`from:7d`) typed on different days cannot collide on a
 * stale cache entry.
 */
export const getPaletteProviderQueryKey = (provider: PaletteSearchProvider, query: PaletteProviderQuery) =>
  [
    'command-palette',
    provider.providerKey,
    provider.contextKey,
    query.text,
    query.range?.from ?? '',
    query.range?.to ?? '',
  ] as const;
