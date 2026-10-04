import { apiFetch, apiFetchJson } from '@platform/transport/http';

/** Keep relationships transport independent so its lazy store does not create another eager shared API chunk. */

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
