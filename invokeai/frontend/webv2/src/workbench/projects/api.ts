import { ApiError, apiFetch, apiFetchJson } from '@workbench/backend/http';

/**
 * REST surface for server-side project persistence (`/api/v1/projects`) and
 * the per-user client-state KV (`/api/v1/client_state`) that holds the small
 * account-scoped workbench blob. Both work in single-user mode too — the
 * backend scopes them to the system user.
 */

const PROJECTS_BASE = '/api/v1/projects';
// The path's queue segment is ignored by the backend (kept for compatibility).
const CLIENT_STATE_BASE = '/api/v1/client_state/default';

export interface ProjectSummaryDTO {
  project_id: string;
  name: string;
  revision: number;
  created_at: string;
  updated_at: string;
}

export interface ProjectRecordDTO extends ProjectSummaryDTO {
  data: Record<string, unknown>;
}

export interface ProjectCreateRequest {
  project_id?: string;
  name: string;
  data: Record<string, unknown>;
}

export interface ProjectUpdateRequest {
  name: string;
  data: Record<string, unknown>;
  expected_revision: number;
}

export const listProjects = (): Promise<ProjectSummaryDTO[]> => apiFetchJson<ProjectSummaryDTO[]>(`${PROJECTS_BASE}/`);

export const getProject = (projectId: string): Promise<ProjectRecordDTO> =>
  apiFetchJson<ProjectRecordDTO>(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}`);

export const createProject = (request: ProjectCreateRequest): Promise<ProjectRecordDTO> =>
  apiFetchJson<ProjectRecordDTO>(`${PROJECTS_BASE}/`, { body: JSON.stringify(request), method: 'POST' });

export const updateProject = (projectId: string, request: ProjectUpdateRequest): Promise<ProjectRecordDTO> =>
  apiFetchJson<ProjectRecordDTO>(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}`, {
    body: JSON.stringify(request),
    method: 'PUT',
  });

export const deleteProject = async (projectId: string): Promise<void> => {
  await apiFetch(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}`, { method: 'DELETE' });
};

/** A save was based on a stale revision — another tab or device saved first. */
export const isProjectConflictError = (error: unknown): boolean => error instanceof ApiError && error.status === 409;

export const isProjectNotFoundError = (error: unknown): boolean => error instanceof ApiError && error.status === 404;

export const getClientStateValue = (key: string): Promise<string | null> =>
  apiFetchJson<string | null>(`${CLIENT_STATE_BASE}/get_by_key?key=${encodeURIComponent(key)}`);

/** The endpoint takes a JSON-encoded string body, hence the stringify of a string. */
export const setClientStateValue = async (key: string, value: string): Promise<void> => {
  await apiFetchJson<string>(`${CLIENT_STATE_BASE}/set_by_key?key=${encodeURIComponent(key)}`, {
    body: JSON.stringify(value),
    method: 'POST',
  });
};

export const deleteClientStateValue = async (key: string): Promise<void> => {
  await apiFetch(`${CLIENT_STATE_BASE}/delete_by_key?key=${encodeURIComponent(key)}`, { method: 'POST' });
};
