import { apiFetch, apiFetchJson, buildApiUrl } from '@platform/transport/http';

export interface ModelsDataPort {
  buildUrl(path: string): string;
  request(path: string, init?: RequestInit): Promise<Response>;
  requestJson<T>(path: string, init?: RequestInit): Promise<T>;
}

export const browserModelsDataPort: ModelsDataPort = {
  buildUrl: buildApiUrl,
  request: apiFetch,
  requestJson: apiFetchJson,
};
