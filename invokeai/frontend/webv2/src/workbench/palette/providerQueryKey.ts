import type { PaletteSearchProvider } from './entries';

export const getPaletteProviderQueryKey = (provider: PaletteSearchProvider, query: string) =>
  ['command-palette', provider.providerKey, provider.contextKey, query] as const;
