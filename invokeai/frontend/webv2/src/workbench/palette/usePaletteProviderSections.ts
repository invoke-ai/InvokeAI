import { useQueries } from '@tanstack/react-query';
import { useMemo } from 'react';

import type { PaletteProviderQuery, PaletteSearchProvider, ProviderResultSection } from './entries';

import { getPaletteProviderQueryKey } from './providerQueryKey';

export interface PaletteProviderQueryResult {
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
