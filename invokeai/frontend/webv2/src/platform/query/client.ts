import { QueryClient } from '@tanstack/react-query';

/** Features own keys/options; React and realtime runtimes share this client. */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5_000,
    },
  },
});
