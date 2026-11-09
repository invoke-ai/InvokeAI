import type { components } from '@api/schema';

import { apiFetchJson } from '@platform/transport/http';
import { queryOptions, useQuery } from '@tanstack/react-query';

/**
 * The deployment's presentation switches. One query key for every reader, so the menus that gate on it share a single
 * request. The values change only with a server restart: they are read once and a failed read stays absent until a
 * menu asks again.
 */
export const frontendConfigQueryOptions = queryOptions({
  queryFn: ({ signal }) =>
    apiFetchJson<components['schemas']['FrontendConfig']>('/api/v1/app/frontend_config', { signal }),
  queryKey: ['frontend-config'],
  staleTime: Infinity,
});

/** True only once the server has said so; unknown and failed reads are false, so the gated entry is absent. */
export const usePreciseImageStorage = (): boolean =>
  useQuery({
    ...frontendConfigQueryOptions,
    select: (config) => config.precise_image_storage === true,
  }).data === true;
