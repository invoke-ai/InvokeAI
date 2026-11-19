import type { components } from '@api/schema';
import type { AuthSession } from '@features/identity';

import { queryClient as appQueryClient } from '@platform/query/client';
import { apiFetchJson } from '@platform/transport/http';
import { type QueryClient, queryOptions } from '@tanstack/react-query';

/** Servers older than a field omit it; a missing permission reads as not granted. */
export type FrontendConfig = Partial<components['schemas']['FrontendConfig']>;

const FRONTEND_CONFIG_KEY = 'frontend-config';

/*
 * A leaf both routes share: the editor's status bar reads it at boot, and the lazily loaded donation link and Server
 * Output view read it too. Callers pass the session, so it imports no Identity runtime and stays in the route-shared
 * chunk (`ROUTE_SHARED_MODULES` in vite.config.mts) instead of costing a startup request of its own.
 */

/**
 * The answer depends on the signed-in account and its role, so every account lifetime and every role the session
 * knows reads its own entry. A role change made elsewhere reaches this tab as a refusal, which invalidates it.
 */
export const frontendConfigQueryKey = (session: AuthSession) =>
  [FRONTEND_CONFIG_KEY, session.accountEpoch, session.user?.user_id ?? null, session.user?.is_admin ?? null] as const;

export const frontendConfigQueryOptions = (session: AuthSession) =>
  queryOptions({
    queryKey: frontendConfigQueryKey(session),
    queryFn: ({ signal }) => apiFetchJson<FrontendConfig>('/api/v1/app/frontend_config', { signal }),
    staleTime: 60_000,
  });

/** The server refused something this config granted, or a refused view asks again: read the grants again. */
export const refetchFrontendConfig = (queryClient: QueryClient = appQueryClient): Promise<void> =>
  queryClient.invalidateQueries({ queryKey: [FRONTEND_CONFIG_KEY] });
