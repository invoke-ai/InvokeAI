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

export const listModels = async (): Promise<ModelConfig[]> => {
  const body = await requestJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/`);

  return body.models ?? [];
};

/** Models whose database record points at a file that no longer exists. */
export const listMissingModels = async (): Promise<ModelConfig[]> => {
  const body = await requestJson<{ models?: ModelConfig[] }>(`${MODELS_BASE}/missing`);

  return body.models ?? [];
};

/** Absolute server path of the models directory; relative model `path`s resolve against it. */
export const getModelsDir = (): Promise<string> => requestJson<string>(`${MODELS_BASE}/models_dir`);

export const getModel = (key: string): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}`);

export const updateModel = (key: string, changes: ModelRecordChanges): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, {
    body: JSON.stringify(changes),
    method: 'PATCH',
  });

export const deleteModel = async (key: string): Promise<void> => {
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}`, { method: 'DELETE' });
};

export const bulkDeleteModels = (keys: string[]): Promise<BulkDeleteModelsResponse> =>
  requestJson<BulkDeleteModelsResponse>(`${MODELS_BASE}/i/bulk_delete`, {
    body: JSON.stringify({ keys }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });

/** Re-probe the model files to refresh auto-detected base/type/format. */
export const reidentifyModel = (key: string): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/i/${encodeURIComponent(key)}/reidentify`, { method: 'POST' });

export const convertModelToDiffusers = (key: string): Promise<ModelConfig> =>
  requestJson<ModelConfig>(`${MODELS_BASE}/convert/${encodeURIComponent(key)}`, { method: 'PUT' });

/** Stable URL for a model's cover image; cache-busted via updated marker. */
export const getModelImageUrl = (key: string, cacheKey?: string): string =>
  buildUrl(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image${cacheKey ? `?t=${encodeURIComponent(cacheKey)}` : ''}`);

export const updateModelImage = async (key: string, image: Blob): Promise<void> => {
  const formData = new FormData();

  formData.append('image', image);
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { body: formData, method: 'PATCH' });
};

export const deleteModelImage = async (key: string): Promise<void> => {
  await request(`${MODELS_BASE}/i/${encodeURIComponent(key)}/image`, { method: 'DELETE' });
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

  return requestJson<ModelInstallJob>(`${MODELS_BASE}/install?${params.toString()}`, {
    body: JSON.stringify(config ?? {}),
    headers,
    method: 'POST',
  });
};

export const listModelInstalls = (): Promise<ModelInstallJob[]> =>
  requestJson<ModelInstallJob[]>(`${MODELS_BASE}/install`);

export const cancelModelInstall = async (id: number): Promise<void> => {
  await request(`${MODELS_BASE}/install/${id}`, { method: 'DELETE' });
};

export const pauseModelInstall = (id: number): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/pause`, { method: 'POST' });

export const resumeModelInstall = (id: number): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/resume`, { method: 'POST' });

export const restartFailedModelInstall = (id: number): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/restart_failed`, { method: 'POST' });

/** Restart a single download part (by its file URL) of an install job. */
export const restartModelInstallFile = (id: number, fileSource: string): Promise<ModelInstallJob> =>
  requestJson<ModelInstallJob>(`${MODELS_BASE}/install/${id}/restart_file`, {
    body: JSON.stringify(fileSource),
    method: 'POST',
  });

export const pruneCompletedModelInstalls = async (): Promise<void> => {
  await request(`${MODELS_BASE}/install`, { method: 'DELETE' });
};

export const scanFolderForModels = (scanPath: string): Promise<FoundModel[]> =>
  requestJson<FoundModel[]>(`${MODELS_BASE}/scan_folder?scan_path=${encodeURIComponent(scanPath)}`);

export const getHuggingFaceModels = (repo: string): Promise<HuggingFaceModels> =>
  requestJson<HuggingFaceModels>(`${MODELS_BASE}/hugging_face?hugging_face_repo=${encodeURIComponent(repo)}`);

export const getStarterModels = (): Promise<StarterModelResponse> =>
  requestJson<StarterModelResponse>(`${MODELS_BASE}/starter_models`);

export const getHFTokenStatus = (): Promise<HFTokenStatus> => requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`);

export const setHFToken = (token: string): Promise<HFTokenStatus> =>
  requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { body: JSON.stringify({ token }), method: 'POST' });

export const resetHFToken = (): Promise<HFTokenStatus> =>
  requestJson<HFTokenStatus>(`${MODELS_BASE}/hf_login`, { method: 'DELETE' });

export const getOrphanedModels = (): Promise<OrphanedModelInfo[]> =>
  requestJson<OrphanedModelInfo[]>(`${MODELS_BASE}/sync/orphaned`);

export const deleteOrphanedModels = (paths: string[]): Promise<DeleteOrphanedModelsResponse> =>
  requestJson<DeleteOrphanedModelsResponse>(`${MODELS_BASE}/sync/orphaned`, {
    body: JSON.stringify({ paths }),
    headers: { 'Content-Type': 'application/json' },
    method: 'DELETE',
  });

export const emptyModelCache = async (): Promise<void> => {
  await request(`${MODELS_BASE}/empty_model_cache`, { method: 'POST' });
};

/** Model relationships: bidirectional "related model" links between configs. */

const RELATIONSHIPS_BASE = '/api/v1/model_relationships';

export const getRelatedModelKeys = (key: string): Promise<string[]> =>
  requestJson<string[]>(`${RELATIONSHIPS_BASE}/i/${encodeURIComponent(key)}`);

export const addModelRelationship = async (modelKey1: string, modelKey2: string): Promise<void> => {
  await request(`${RELATIONSHIPS_BASE}/`, {
    body: JSON.stringify({ model_key_1: modelKey1, model_key_2: modelKey2 }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });
};

export const removeModelRelationship = async (modelKey1: string, modelKey2: string): Promise<void> => {
  await request(`${RELATIONSHIPS_BASE}/`, {
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
  requestJson<ExternalProviderConfig[]>(`${EXTERNAL_PROVIDERS_BASE}/config`);

export const setExternalProviderConfig = (
  providerId: string,
  config: { api_key?: string; base_url?: string | null }
): Promise<ExternalProviderConfig> =>
  requestJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    body: JSON.stringify(config),
    method: 'POST',
  });

export const resetExternalProviderConfig = (providerId: string): Promise<ExternalProviderConfig> =>
  requestJson<ExternalProviderConfig>(`${EXTERNAL_PROVIDERS_BASE}/config/${encodeURIComponent(providerId)}`, {
    method: 'DELETE',
  });
