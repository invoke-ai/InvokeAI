import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

import type { PaletteProviderQuery, PaletteSearchProvider, ProviderResultSection } from './entries';

import { PaletteSearchUnavailableError } from './entries';
import { getPaletteProviderQueryKey } from './providerQueryKey';

export interface PaletteProviderQueryResult {
  /** A failure's user-facing explanation, when the provider gave one. */
  errorMessage: string | null;
  isError: boolean;
  isFetching: boolean;
  retry: () => void;
}

export const usePaletteProviderSections = ({
  enabled,
  isWaitingForDebounce,
  providerQuery,
  providers,
}: {
  enabled: boolean;
  isWaitingForDebounce: boolean;
  providerQuery: PaletteProviderQuery;
  providers: PaletteSearchProvider[];
}): { results: PaletteProviderQueryResult[]; sections: ProviderResultSection[] } => {
  const queryResults = useQueries({
    combine: (results) =>
      results.map((result) => ({
        data: result.data,
        errorMessage: result.error instanceof PaletteSearchUnavailableError ? result.error.message : null,
        isError: result.isError,
        isFetching: result.isFetching,
        retry: () => void result.refetch(),
      })),
    queries: providers.map((provider) => ({
      enabled,
      gcTime: 60_000,
      queryFn: ({ signal }: { signal: AbortSignal }) => Promise.resolve(provider.search(providerQuery, { signal })),
      queryKey: getPaletteProviderQueryKey(provider, providerQuery),
      retry: false,
      staleTime: 0,
    })),
  });

  const results = useMemo<PaletteProviderQueryResult[]>(
    () =>
      queryResults.map((result) => ({
        errorMessage: result.errorMessage,
        isError: result.isError,
        isFetching: result.isFetching,
        retry: result.retry,
      })),
    [queryResults]
  );
  const sections = useMemo<ProviderResultSection[]>(
    () =>
      providers.map((provider, index) => ({
        entries: queryResults[index]?.data ?? [],
        isError: queryResults[index]?.isError ?? false,
        isFetching: enabled && (queryResults[index]?.isFetching ?? false),
        isWaitingForDebounce,
        provider,
        retry: queryResults[index]?.retry ?? (() => undefined),
      })),
    [enabled, isWaitingForDebounce, providers, queryResults]
  );

  return { results, sections };
};
