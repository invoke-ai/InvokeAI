import type {
  BulkDeleteModelsResponse,
  DeleteOrphanedModelsResponse,
  FoundModel,
  HFTokenStatus,
  HuggingFaceModels,
  ModelConfig,
  ModelInstallJob,
  ModelRecordChanges,
  OrphanedModelInfo,
  StarterModelResponse,
} from '@features/models/core/types';

import { browserModelsDataPort } from './transport';

/** REST client for the backend model manager (`/api/v2/models`). */

const MODELS_BASE = '/api/v2/models';
const { buildUrl, request, requestJson } = browserModelsDataPort;

export const listModels = async (signal?: AbortSignal): Promise<ModelConfig[]> => {
  const body = await requestJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/`, { signal });

  return body.models ?? [];
};

/** Models whose database record points at a file that no longer exists. */
export const listMissingModels = async (signal?: AbortSignal): Promise<ModelConfig[]> => {
  const body = await requestJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/missing`, { signal });

  return body.models ?? [];
};

/** Absolute server path of the models directory; relative model `path`s resolve against it. */
export const getModelsDir = (signal?: AbortSignal): Promise<string> =>
  requestJson<string>(`${MODELS_BASE}/models_dir`, { signal });

export const updateModel = (key: string, changes: ModelRecordChanges, signal?: AbortSignal): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, {
    body: JSON.stringify(changes),
    method: 'PATCH',
    signal,
  });

export const deleteModel = async (key: string, signal?: AbortSignal): Promise<void> => {
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, { method: 'DELETE', signal });
};

export const bulkDeleteModels = (keys: string[], signal?: AbortSignal): Promise<BulkDeleteModelsResponse> =>
  requestJson<BulkDeleteModelsResponse>(`${MODELS_BASE}/i/bulk_delete`, {
    body: JSON.stringify({ keys }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    signal,
  });

/** Re-probe the model files to refresh auto-detected base/type/format. */
export const reidentifyModel = (key: string, signal?: AbortSignal): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}/reidentify`, { method: 'POST', signal });

export const convertModelToDiffusers = (key: string, signal?: AbortSignal): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/convert/${encodeURIComponent(key)}`, { method: 'PUT', signal });

/** Stable URL for a model's cover image; cache-busted via updated marker. */
export const getModelImageUrl = (key: string, cacheKey?: string): string =>
  buildUrl(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image${cacheKey ? `?t=${encodeURIComponent(cacheKey)}` : ''}`);

export const updateModelImage = async (key: string, image: Blob, signal?: AbortSignal): Promise<void> => {
  const formData = new FormData();

  formData.append('image', image);
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { body: formData, method: 'PATCH', signal });
};

export const deleteModelImage = async (key: string, signal?: AbortSignal): Promise<void> => {
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { method: 'DELETE', signal });
};

export interface InstallModelRequest {
  source: string;
  inplace?: boolean;
  /** Access token for protected remote sources (e.g. Civitai API key). */
  accessToken?: string;
  /** Overrides for auto-probed config fields (name, base, type, ...). */
  config?: ModelRecordChanges;
}

export const installModel = (
  { accessToken, config, inplace, source }: InstallModelRequest,
  signal?: AbortSignal
): Promise<ModelInstallJob> => {
  const params = new URLSearchParams({ source });
  const headers: HeadersInit = {};

  if (inplace !== undefined) {
    params.set('inplace', String(inplace));
  }

  if (accessToken) {
    headers['X-Model-Source-Access-Token'] = accessToken;
  }

  return requestJson<ModelInstallJob>(`${MODELS_BASE}/install?${params.toString()}`, {
    body: JSON.stringify(config ?? {}),
    headers,
    method: 'POST',
    signal,
  });
};

export const listModelInstalls = (signal?: AbortSignal): Promise<ModelInstallJob[]> =>
  requestJson<ModelInstallJob[]>(`${MODELS_BASE}/install`, { signal });

export const cancelModelInstall = async (id: number, signal?: AbortSignal): Promise<void> => {
  await request(`${MODELS_BASE}/install/${id}`, { method: 'DELETE', signal });
};

export const pauseModelInstall = (id: number, signal?: AbortSignal): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/pause`, { method: 'POST', signal });

export const resumeModelInstall = (id: number, signal?: AbortSignal): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/resume`, { method: 'POST', signal });

export const restartFailedModelInstall = (id: number, signal?: AbortSignal): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/restart_failed`, { method: 'POST', signal });

/** Restart a single download part (by its file URL) of an install job. */
export const restartModelInstallFile = (
  id: number,
  fileSource: string,
  signal?: AbortSignal
): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/restart_file`, {
    body: JSON.stringify(fileSource),
    method: 'POST',
    signal,
  });

export const pruneCompletedModelInstalls = async (signal?: AbortSignal): Promise<void> => {
  await request(`${MODELS_BASE}/install`, { method: 'DELETE', signal });
};

export const scanFolderForModels = (scanPath: string, signal?: AbortSignal): Promise<FoundModel[]> =>
  requestJson<FoundModel[]>(`${MODELS_BASE}/scan_folder?scan_path=${encodeURIComponent(scanPath)}`, { signal });

export const getHuggingFaceModels = (repo: string, signal?: AbortSignal): Promise<HuggingFaceModels> =>
  requestJson<HuggingFaceModels>(`${MODELS_BASE}/hugging_face?hugging_face_repo=${encodeURIComponent(repo)}`, {
    signal,
  });

export const getStarterModels = (signal?: AbortSignal): Promise<StarterModelResponse> =>
  requestJson<StarterModelResponse>(`${MODELS_BASE}/starter_models`, { signal });

export const getHFTokenStatus = (signal?: AbortSignal): Promise<HFTokenStatus> =>
  requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { signal });

export const setHFToken = (token: string, signal?: AbortSignal): Promise<HFTokenStatus> =>
  requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, {
    body: JSON.stringify({ token }),
    method: 'POST',
    signal,
  });

export const resetHFToken = (signal?: AbortSignal): Promise<HFTokenStatus> =>
  requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { method: 'DELETE', signal });

export const getOrphanedModels = (signal?: AbortSignal): Promise<OrphanedModelInfo[]> =>
  requestJson<OrphanedModelInfo[]>(`${MODELS_BASE}/sync/orphaned`, { signal });

export const deleteOrphanedModels = (paths: string[], signal?: AbortSignal): Promise<DeleteOrphanedModelsResponse> =>
  requestJson<DeleteOrphanedModelsResponse>(`${MODELS_BASE}/sync/orphaned`, {
    body: JSON.stringify({ paths }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
    signal,
  });

export const emptyModelCache = async (signal?: AbortSignal): Promise<void> => {
  await request(`${MODELS_BASE}/empty_model_cache`, { method: 'POST', signal });
};

/** Model relationships: bidirectional "related model" links between configs. */

const RELATIONSHIPS_BASE = '/api/v1/model_relationships';

export const getRelatedModelKeys = (key: string, signal?: AbortSignal): Promise<string[]> =>
  requestJson<string[]>(`${RELATIONSHIPS_BASE}/i/${encodeURIComponent(key)}`, { signal });

export const addModelRelationship = async (
  modelKey1: string,
  modelKey2: string,
  signal?: AbortSignal
): Promise<void> => {
  await request(`${RELATIONSHIPS_BASE}/`, {
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
  await request(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
    signal,
  });
};

/** External image-provider credentials (OpenAI, Gemini, ...). */

const EXTERNAL_PROVIDERS_BASE = '/api/v1/app/external_providers';

export interface ExternalProviderConfig {
  provider_id: string;
  api_key_configured: boolean;
  base_url: string | null;
}

export const getExternalProviderConfigs = (signal?: AbortSignal): Promise<ExternalProviderConfig[]> =>
  requestJson<ExternalProviderConfig[]>(`${EXTERNAL_PROVIDERS_BASE}/config`, { signal });

export const setExternalProviderConfig = (
  providerId: string,
  config: { api_key?: string; base_url?: string | null },
  signal?: AbortSignal
): Promise<ExternalProviderConfig> =>
  requestJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    body: JSON.stringify(config),
    method: 'POST',
    signal,
  });

export const resetExternalProviderConfig = (
  providerId: string,
  signal?: AbortSignal
): Promise<ExternalProviderConfig> =>
  requestJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    method: 'DELETE',
    signal,
  });
