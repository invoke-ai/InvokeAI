import { apiFetch, apiFetchJson, buildApiUrl } from '../backend/http';
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
} from './types';

/** REST client for the backend model manager (`/api/v2/models`). */

const MODELS_BASE = '/api/v2/models';

export const listModels = async (): Promise<ModelConfig[]> => {
  const body = await apiFetchJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/`);

  return body.models ?? [];
};

/** Models whose database record points at a file that no longer exists. */
export const listMissingModels = async (): Promise<ModelConfig[]> => {
  const body = await apiFetchJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/missing`);

  return body.models ?? [];
};

/** Absolute server path of the models directory; relative model `path`s resolve against it. */
export const getModelsDir = (): Promise<string> => apiFetchJson<string>(`${MODELS_BASE}/models_dir`);

export const getModel = (key: string): Promise<ModelConfig> =>
  apiFetchJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}`);

export const updateModel = (key: string, changes: ModelRecordChanges): Promise<ModelConfig> =>
  apiFetchJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, {
    body: JSON.stringify(changes),
    method: 'PATCH',
  });

export const deleteModel = async (key: string): Promise<void> => {
  await apiFetch(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, { method: 'DELETE' });
};

export const bulkDeleteModels = (keys: string[]): Promise<BulkDeleteModelsResponse> =>
  apiFetchJson<BulkDeleteModelsResponse>(`${MODELS_BASE}/i/bulk_delete`, {
    body: JSON.stringify({ keys }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });

/** Re-probe the model files to refresh auto-detected base/type/format. */
export const reidentifyModel = (key: string): Promise<ModelConfig> =>
  apiFetchJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}/reidentify`, { method: 'POST' });

export const convertModelToDiffusers = (key: string): Promise<ModelConfig> =>
  apiFetchJson<ModelConfig>(`${MODELS_BASE}/convert/${encodeURIComponent(key)}`, { method: 'PUT' });

/** Stable URL for a model's cover image; cache-busted via updated marker. */
export const getModelImageUrl = (key: string, cacheKey?: string): string =>
  buildApiUrl(
    `${MODELS_BASE}/i/${encodeURIComponent(key)}/image${cacheKey ? `?t=${encodeURIComponent(cacheKey)}` : ''}`
  );

export const updateModelImage = async (key: string, image: Blob): Promise<void> => {
  const formData = new FormData();

  formData.append('image', image);
  await apiFetch(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { body: formData, method: 'PATCH' });
};

export const deleteModelImage = async (key: string): Promise<void> => {
  await apiFetch(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { method: 'DELETE' });
};

export interface InstallModelRequest {
  source: string;
  inplace?: boolean;
  /** Access token for protected remote sources (e.g. Civitai API key). */
  accessToken?: string;
  /** Overrides for auto-probed config fields (name, base, type, ...). */
  config?: ModelRecordChanges;
}

export const installModel = ({
  accessToken,
  config,
  inplace,
  source,
}: InstallModelRequest): Promise<ModelInstallJob> => {
  const params = new URLSearchParams({ source });
  const headers: HeadersInit = {};

  if (inplace !== undefined) {
    params.set('inplace', String(inplace));
  }

  if (accessToken) {
    headers['X-Model-Source-Access-Token'] = accessToken;
  }

  return apiFetchJson<ModelInstallJob>(`${MODELS_BASE}/install?${params.toString()}`, {
    body: JSON.stringify(config ?? {}),
    headers,
    method: 'POST',
  });
};

export const listModelInstalls = (): Promise<ModelInstallJob[]> =>
  apiFetchJson<ModelInstallJob[]>(`${MODELS_BASE}/install`);

export const cancelModelInstall = async (id: number): Promise<void> => {
  await apiFetch(`${MODELS_BASE}/install/${id}`, { method: 'DELETE' });
};

export const pauseModelInstall = (id: number): Promise<ModelInstallJob> =>
  apiFetchJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/pause`, { method: 'POST' });

export const resumeModelInstall = (id: number): Promise<ModelInstallJob> =>
  apiFetchJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/resume`, { method: 'POST' });

export const restartFailedModelInstall = (id: number): Promise<ModelInstallJob> =>
  apiFetchJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/restart_failed`, { method: 'POST' });

export const pruneCompletedModelInstalls = async (): Promise<void> => {
  await apiFetch(`${MODELS_BASE}/install`, { method: 'DELETE' });
};

export const scanFolderForModels = (scanPath: string): Promise<FoundModel[]> =>
  apiFetchJson<FoundModel[]>(`${MODELS_BASE}/scan_folder?scan_path=${encodeURIComponent(scanPath)}`);

export const getHuggingFaceModels = (repo: string): Promise<HuggingFaceModels> =>
  apiFetchJson<HuggingFaceModels>(`${MODELS_BASE}/hugging_face?hugging_face_repo=${encodeURIComponent(repo)}`);

export const getStarterModels = (): Promise<StarterModelResponse> =>
  apiFetchJson<StarterModelResponse>(`${MODELS_BASE}/starter_models`);

export const getHFTokenStatus = (): Promise<HFTokenStatus> => apiFetchJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`);

export const setHFToken = (token: string): Promise<HFTokenStatus> =>
  apiFetchJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { body: JSON.stringify({ token }), method: 'POST' });

export const resetHFToken = (): Promise<HFTokenStatus> =>
  apiFetchJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { method: 'DELETE' });

export const getOrphanedModels = (): Promise<OrphanedModelInfo[]> =>
  apiFetchJson<OrphanedModelInfo[]>(`${MODELS_BASE}/sync/orphaned`);

export const deleteOrphanedModels = (paths: string[]): Promise<DeleteOrphanedModelsResponse> =>
  apiFetchJson<DeleteOrphanedModelsResponse>(`${MODELS_BASE}/sync/orphaned`, {
    body: JSON.stringify({ paths }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
  });

export const emptyModelCache = async (): Promise<void> => {
  await apiFetch(`${MODELS_BASE}/empty_model_cache`, { method: 'POST' });
};

/** Model relationships: bidirectional "related model" links between configs. */

const RELATIONSHIPS_BASE = '/api/v1/model_relationships';

export const getRelatedModelKeys = (key: string): Promise<string[]> =>
  apiFetchJson<string[]>(`${RELATIONSHIPS_BASE}/i/${encodeURIComponent(key)}`);

export const addModelRelationship = async (modelKey1: string, modelKey2: string): Promise<void> => {
  await apiFetch(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });
};

export const removeModelRelationship = async (modelKey1: string, modelKey2: string): Promise<void> => {
  await apiFetch(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
  });
};

/** External image-provider credentials (OpenAI, Gemini, ...). */

const EXTERNAL_PROVIDERS_BASE = '/api/v1/app/external_providers';

export interface ExternalProviderConfig {
  provider_id: string;
  api_key_configured: boolean;
  base_url: string | null;
}

export const getExternalProviderConfigs = (): Promise<ExternalProviderConfig[]> =>
  apiFetchJson<ExternalProviderConfig[]>(`${EXTERNAL_PROVIDERS_BASE}/config`);

export const setExternalProviderConfig = (
  providerId: string,
  config: { api_key?: string; base_url?: string | null }
): Promise<ExternalProviderConfig> =>
  apiFetchJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    body: JSON.stringify(config),
    method: 'POST',
  });

export const resetExternalProviderConfig = (providerId: string): Promise<ExternalProviderConfig> =>
  apiFetchJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    method: 'DELETE',
  });
