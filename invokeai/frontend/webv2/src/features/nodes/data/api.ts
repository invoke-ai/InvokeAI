import { normalizeNodePackCatalog, type NodePackCatalog } from '@features/nodes/core/catalog';

import { browserNodesDataPort } from './transport';

/** REST client for the custom nodes manager (`/api/v2/custom_nodes`). */

const CUSTOM_NODES_BASE = '/api/v2/custom_nodes';
const { requestJson } = browserNodesDataPort;

export interface InstallNodePackResponse {
  name: string;
  success: boolean;
  message: string;
  workflows_imported: number;
  requires_dependencies: boolean;
  dependency_file: string | null;
}

export interface UninstallNodePackResponse {
  name: string;
  success: boolean;
  message: string;
}

export const listCustomNodePacks = async (signal?: AbortSignal): Promise<NodePackCatalog> =>
  normalizeNodePackCatalog(await requestJson<unknown>(`${CUSTOM_NODES_BASE}/`, { signal }));

export const installCustomNodePack = (source: string, signal?: AbortSignal): Promise<InstallNodePackResponse> =>
  requestJson<InstallNodePackResponse>(`${CUSTOM_NODES_BASE}/install`, {
    body: JSON.stringify({ source }),
    method: 'POST',
    signal,
  });

export const uninstallCustomNodePack = (packName: string, signal?: AbortSignal): Promise<UninstallNodePackResponse> =>
  requestJson<UninstallNodePackResponse>(`${CUSTOM_NODES_BASE}/${encodeURIComponent(packName)}`, {
    method: 'DELETE',
    signal,
  });

/** The body's status is prose (e.g. "No custom nodes directory found.") — callers must not assume success. */
export const reloadCustomNodes = (signal?: AbortSignal): Promise<{ status: string }> =>
  requestJson<{ status: string }>(`${CUSTOM_NODES_BASE}/reload`, { method: 'POST', signal });
