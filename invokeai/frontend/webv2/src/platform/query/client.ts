import { QueryClient } from '@tanstack/react-query';

/**
 * The application query cache. Feature modules own their keys and options;
 * non-React runtimes use this same client for precise realtime updates.
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5_000,
    },
  },
});
