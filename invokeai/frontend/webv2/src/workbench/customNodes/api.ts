import { apiFetch, apiFetchJson } from '@workbench/backend/http';

/** REST client for the custom nodes manager (`/api/v2/custom_nodes`). */

const CUSTOM_NODES_BASE = '/api/v2/custom_nodes';

export interface NodePackInfo {
  name: string;
  path: string;
  node_count: number;
  node_types: string[];
}

export interface NodePackListResponse {
  node_packs: NodePackInfo[];
  custom_nodes_path: string;
}

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

export const listCustomNodePacks = (): Promise<NodePackListResponse> =>
  apiFetchJson<NodePackListResponse>(`${CUSTOM_NODES_BASE}/`);

export const installCustomNodePack = (source: string): Promise<InstallNodePackResponse> =>
  apiFetchJson<InstallNodePackResponse>(`${CUSTOM_NODES_BASE}/install`, {
    body: JSON.stringify({ source }),
    method: 'POST',
  });

export const uninstallCustomNodePack = (packName: string): Promise<UninstallNodePackResponse> =>
  apiFetchJson<UninstallNodePackResponse>(`${CUSTOM_NODES_BASE}/${encodeURIComponent(packName)}`, { method: 'DELETE' });

export const reloadCustomNodes = async (): Promise<void> => {
  await apiFetch(`${CUSTOM_NODES_BASE}/reload`, { method: 'POST' });
};
