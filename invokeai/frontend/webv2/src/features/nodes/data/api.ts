import { normalizeNodePackCatalog, type NodePackCatalog } from '@features/nodes/core/catalog';

import { browserNodesDataPort } from './transport';

/** REST client for the custom nodes manager (`/api/v2/custom_nodes`). */

const CUSTOM_NODES_BASE = '/api/v2/custom_nodes';
const { request, requestJson } = browserNodesDataPort;

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

export const listCustomNodePacks = async (): Promise<NodePackCatalog> =>
  normalizeNodePackCatalog(await requestJson<unknown>(`${CUSTOM_NODES_BASE}/`));

export const installCustomNodePack = (source: string): Promise<InstallNodePackResponse> =>
  requestJson<InstallNodePackResponse>(`${CUSTOM_NODES_BASE}/install`, {
    body: JSON.stringify({ source }),
    method: 'POST',
  });

export const uninstallCustomNodePack = (packName: string): Promise<UninstallNodePackResponse> =>
  requestJson<UninstallNodePackResponse>(`${CUSTOM_NODES_BASE}/${encodeURIComponent(packName)}`, { method: 'DELETE' });

export const reloadCustomNodes = async (): Promise<void> => {
  await request(`${CUSTOM_NODES_BASE}/reload`, { method: 'POST' });
};
