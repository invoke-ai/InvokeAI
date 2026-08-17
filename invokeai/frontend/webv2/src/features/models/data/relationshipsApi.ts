import { apiFetch, apiFetchJson } from '@platform/transport/http';

/**
 * REST client for `/api/v1/model_relationships`, kept separate from `api.ts`
 * so the lazily-loaded relationships store is a self-contained chunk: sharing
 * the models api module with the eager modelsStore would split that module
 * into an extra eagerly-fetched chunk (the initial-request budget).
 */

const RELATIONSHIPS_BASE = '/api/v1/model_relationships';

export const getRelatedModelKeys = (key: string, signal?: AbortSignal): Promise<string[]> =>
  apiFetchJson<string[]>(`${RELATIONSHIPS_BASE}/i/${encodeURIComponent(key)}`, { signal });

export const addModelRelationship = async (
  modelKey1: string,
  modelKey2: string,
  signal?: AbortSignal
): Promise<void> => {
  await apiFetch(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    signal,
  });
};

export const removeModelRelationship = async (
  modelKey1: string,
  modelKey2: string,
  signal?: AbortSignal
): Promise<void> => {
  await apiFetch(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
    signal,
  });
};
