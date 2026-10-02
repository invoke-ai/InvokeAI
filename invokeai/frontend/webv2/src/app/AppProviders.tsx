import type { ReactNode } from 'react';

import { queryClient } from '@platform/query/client';
import { QueryClientProvider } from '@tanstack/react-query';

/** Application-wide infrastructure providers shared by every route. */
export const AppProviders = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
);
