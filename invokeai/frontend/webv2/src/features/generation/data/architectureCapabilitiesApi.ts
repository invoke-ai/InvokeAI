import type { ArchitectureCapabilitiesRow } from '@features/generation/core/architectureCapabilities';

import { apiFetchJson } from '@platform/transport/http';

/** The architecture table is static per backend build and independent of installed models. */
const CAPABILITIES_PATH = '/api/v2/models/capabilities';

export const getArchitectureCapabilities = (signal?: AbortSignal): Promise<ArchitectureCapabilitiesRow[]> =>
  apiFetchJson<ArchitectureCapabilitiesRow[]>(CAPABILITIES_PATH, { signal });
