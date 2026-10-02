import { apiFetch, apiFetchJson, buildApiUrl } from '@platform/transport/http';

export interface NodesDataPort {
  buildUrl(path: string): string;
  request(path: string, init?: RequestInit): Promise<Response>;
  requestJson<T>(path: string, init?: RequestInit): Promise<T>;
}

export const browserNodesDataPort: NodesDataPort = {
  buildUrl: buildApiUrl,
  request: apiFetch,
  requestJson: apiFetchJson,
};
