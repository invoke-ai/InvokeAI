import type { AccountScope } from '@platform/state/accountLifecycle';

import { queryClient } from '@platform/query/client';
import { assertAccountScopeCurrent, captureAccountScope } from '@platform/state/accountLifecycle';
import {
  ApiError,
  apiFetch,
  apiFetchJson,
  RequestDeadlineError,
  sleep,
  withRequestDeadline,
} from '@platform/transport/http';
import {
  DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
  MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
} from '@workbench/canvasSchemaVersion';

import { PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION } from './documentSchemaVersion';

/** Project records and client state are scoped to the authenticated account. */

const PROJECTS_BASE = '/api/v1/projects';
const PROJECT_WRITE_MAX_ATTEMPTS = 9;
const PROJECT_WRITE_RETRY_WINDOW_MS = 125_000;
// The server permits 120 seconds to receive a project body; allow another 30 for processing and the response.
const PROJECT_REQUEST_TIMEOUT_MS = 150_000;
const CLIENT_STATE_READ_TIMEOUT_MS = 30_000;
// The path's queue segment is ignored by the backend (kept for compatibility).
const CLIENT_STATE_BASE = '/api/v1/client_state/default';

export interface ProjectSummaryDTO {
  project_id: string;
  /** The server's board ID is authoritative; the document stores only a cache. */
  board_id: string;
  name: string;
  revision: number;
  minimum_canvas_schema_version: number;
  /** The oldest project document schema a client must read; 3 for a project saved whole, 4 once split. */
  minimum_document_schema_version: number;
  created_at: string;
  updated_at: string;
  workspace_updated_at: string | null;
}

export interface ProjectWorkspaceDTO {
  data: Record<string, unknown>;
  revision: number;
  updated_at: string;
}

export interface ProjectRecordDTO extends ProjectSummaryDTO {
  data: Record<string, unknown>;
}

/** A read, create or update answer: the workspace beside the content, null when there is none. */
export interface ProjectWithWorkspaceDTO extends ProjectRecordDTO {
  workspace: ProjectWorkspaceDTO | null;
}

export interface ProjectCreateRequest {
  project_id?: string;
  /** Adopt and rename an unclaimed private board, or omit to create one atomically with the project. */
  board_id?: string;
  name: string;
  data: Record<string, unknown>;
  /** Raise the server's compatibility floor atomically with this document. */
  minimum_canvas_schema_version?: number;
  /** 4 when `data` is project content, which every save of a split project is; omitted for a document saved whole. */
  minimum_document_schema_version?: number;
  /** Stored in the same transaction as the content; only beside split content (document floor 4). */
  workspace?: Record<string, unknown>;
}

export type ProjectBoardItemKind = 'image' | 'video';
export type ProjectBoardItemCategory = 'general' | 'control' | 'mask' | 'user';

export interface ProjectBoardItemDTO {
  category: ProjectBoardItemCategory;
  kind: ProjectBoardItemKind;
  name: string;
  starred: boolean;
}

export interface ProjectBoardSnapshotBoardDTO {
  archived: boolean;
  board_id: string;
  is_inbox: boolean;
  items: ProjectBoardItemDTO[];
  name: string;
}

export interface ProjectBoardSnapshotDTO {
  boards: ProjectBoardSnapshotBoardDTO[];
}

export interface ProjectUpdateRequest {
  name: string;
  data: Record<string, unknown>;
  expected_revision: number;
  /** Raise the compatibility floor atomically with this save; omission preserves the stored floor. */
  minimum_canvas_schema_version?: number;
  /** 4 when `data` is project content, which splits a project saved whole; omission preserves the stored floor. */
  minimum_document_schema_version?: number;
  /** Same transaction as the content, so a whole project splits atomically; a revision mismatch refuses both. */
  workspace?: ProjectWorkspaceSaveRequest | null;
}

export interface ProjectWorkspaceSaveRequest {
  data: Record<string, unknown>;
  expected_revision: number | null;
}

export const listProjects = (signal?: AbortSignal): Promise<ProjectSummaryDTO[]> =>
  withRequestDeadline(
    (signal) => apiFetchJson<ProjectSummaryDTO[]>(`${PROJECTS_BASE}/`, { signal }),
    PROJECT_REQUEST_TIMEOUT_MS,
    signal
  );

const getProjectPath = (projectId: string): string =>
  `${PROJECTS_BASE}/${encodeURIComponent(projectId)}?max_canvas_schema_version=${MAX_SUPPORTED_CANVAS_SCHEMA_VERSION}` +
  `&max_document_schema_version=${PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION}`;

export const getProject = (projectId: string, signal?: AbortSignal): Promise<ProjectWithWorkspaceDTO> =>
  withRequestDeadline(
    (signal) => apiFetchJson<ProjectWithWorkspaceDTO>(getProjectPath(projectId), { signal }),
    PROJECT_REQUEST_TIMEOUT_MS,
    signal
  );

/**
 * Null without parsing past `maxBytes` decoded. `Content-Length` counts compressed bytes, so the body is streamed and
 * counted rather than trusted.
 */
export const getProjectWithinSize = async (
  projectId: string,
  maxBytes: number,
  signal?: AbortSignal
): Promise<ProjectWithWorkspaceDTO | null> => {
  const response = await apiFetch(getProjectPath(projectId), { signal });
  const declared = Number(response.headers.get('Content-Length'));

  if (Number.isFinite(declared) && declared > maxBytes) {
    await response.body?.cancel();
    return null;
  }

  const decoder = new TextDecoder();
  let text = '';
  let size = 0;

  if (response.body) {
    const reader = response.body.getReader();

    for (;;) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }
      size += value.byteLength;
      if (size > maxBytes) {
        await reader.cancel();
        return null;
      }
      text += decoder.decode(value, { stream: true });
    }
  }

  return JSON.parse(text + decoder.decode()) as ProjectWithWorkspaceDTO;
};

const isProjectWriteBusyError = (error: unknown): error is ApiError => {
  if (!(error instanceof ApiError) || error.status !== 429) {
    return false;
  }
  try {
    return (JSON.parse(error.message) as { detail?: { code?: unknown } }).detail?.code === 'project_write_busy';
  } catch {
    return false;
  }
};

const getRetryDelayMs = (error: ApiError): number => {
  const raw = error.headers.get('Retry-After');
  if (raw === null) {
    return 1_000;
  }
  const seconds = Number(raw);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1_000;
  }
  const date = Date.parse(raw);
  return Number.isFinite(date) ? Math.max(date - Date.now(), 0) : 1_000;
};

interface ProjectWriteRetryBudget {
  attempts: number;
  readonly deadline: number;
}

const createProjectWriteRetryBudget = (): ProjectWriteRetryBudget => ({
  attempts: 0,
  deadline: Date.now() + PROJECT_WRITE_RETRY_WINDOW_MS,
});

const requestProjectWrite = async <T>(
  request: (signal: AbortSignal) => Promise<T>,
  signal: AbortSignal | undefined,
  budget: ProjectWriteRetryBudget = createProjectWriteRetryBudget()
): Promise<T> => {
  const requestSignal = AbortSignal.any([captureAccountScope().signal, ...(signal ? [signal] : [])]);
  while (budget.attempts < PROJECT_WRITE_MAX_ATTEMPTS) {
    budget.attempts += 1;
    try {
      requestSignal.throwIfAborted();
      if (Date.now() >= budget.deadline) {
        throw new RequestDeadlineError();
      }
      return await withRequestDeadline(request, PROJECT_REQUEST_TIMEOUT_MS, requestSignal);
    } catch (error) {
      if (!isProjectWriteBusyError(error) || budget.attempts >= PROJECT_WRITE_MAX_ATTEMPTS) {
        throw error;
      }
      const retryAfterMs = getRetryDelayMs(error);
      const backoffCeilingMs = Math.min(1_000 * 2 ** (budget.attempts - 1), 30_000);
      const remainingMs = budget.deadline - Date.now();
      const delayMs =
        budget.attempts === PROJECT_WRITE_MAX_ATTEMPTS - 1
          ? Math.max(retryAfterMs, remainingMs - 1_000)
          : retryAfterMs + Math.random() * backoffCeilingMs;

      if (delayMs > remainingMs) {
        throw error;
      }
      await sleep(delayMs, requestSignal);
    }
  }

  throw new Error('Project write retry budget exhausted');
};

const serializeCreateProjectRequest = (request: ProjectCreateRequest): string =>
  JSON.stringify({
    ...request,
    max_canvas_schema_version: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
    max_document_schema_version: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
    minimum_canvas_schema_version: request.minimum_canvas_schema_version ?? DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION,
  });

/** Judged by the content: a committed create stored its workspace with it, so the record found carries it too. */
const projectRecordMatchesCreate = (record: ProjectRecordDTO, requestBody: string): boolean => {
  const request = JSON.parse(requestBody) as ProjectCreateRequest;

  return (
    record.project_id === request.project_id &&
    record.name === request.name &&
    JSON.stringify(record.data) === JSON.stringify(request.data) &&
    (request.board_id === undefined || record.board_id === request.board_id) &&
    record.minimum_canvas_schema_version >=
      (request.minimum_canvas_schema_version ?? DEFAULT_PROJECT_CANVAS_SCHEMA_VERSION) &&
    record.minimum_document_schema_version >= (request.minimum_document_schema_version ?? 1)
  );
};

const createProjectFromBody = (
  body: string,
  signal: AbortSignal | undefined,
  budget: ProjectWriteRetryBudget
): Promise<ProjectWithWorkspaceDTO> =>
  requestProjectWrite(
    (signal) =>
      apiFetchJson<ProjectWithWorkspaceDTO>(`${PROJECTS_BASE}/`, {
        body,
        method: 'POST',
        signal,
      }),
    signal,
    budget
  );

export const createProject = (request: ProjectCreateRequest, signal?: AbortSignal): Promise<ProjectWithWorkspaceDTO> =>
  createProjectFromBody(serializeCreateProjectRequest(request), signal, createProjectWriteRetryBudget());

export const updateProject = (
  projectId: string,
  request: ProjectUpdateRequest,
  signal?: AbortSignal
): Promise<ProjectWithWorkspaceDTO> => {
  const body = JSON.stringify({
    ...request,
    max_canvas_schema_version: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
    max_document_schema_version: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
  });

  return requestProjectWrite(
    (signal) =>
      apiFetchJson<ProjectWithWorkspaceDTO>(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}`, {
        body,
        method: 'PUT',
        signal,
      }),
    signal,
    createProjectWriteRetryBudget()
  );
};

/** Declares the schemas this build reads: a project whose canvas floor is above them is refused. */
export const saveProjectWorkspace = (
  projectId: string,
  request: ProjectWorkspaceSaveRequest,
  signal?: AbortSignal
): Promise<ProjectWorkspaceDTO> => {
  const body = JSON.stringify({
    ...request,
    max_canvas_schema_version: MAX_SUPPORTED_CANVAS_SCHEMA_VERSION,
    max_document_schema_version: PROJECT_SPLIT_DOCUMENT_SCHEMA_VERSION,
  });

  return requestProjectWrite(
    (signal) =>
      apiFetchJson<ProjectWorkspaceDTO>(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}/workspace`, {
        body,
        method: 'PUT',
        signal,
      }),
    signal,
    createProjectWriteRetryBudget()
  );
};

export type DeleteProjectBoards = 'delete' | 'release';

/** Media is never deleted: released boards keep theirs in the Library, deleted boards' returns to Uncategorized. */
export const deleteProject = async (
  projectId: string,
  signal?: AbortSignal,
  boards: DeleteProjectBoards = 'release'
): Promise<void> => {
  signal ??= captureAccountScope().signal;
  const query = boards === 'release' ? '' : '?boards=delete';

  try {
    await withRequestDeadline(
      (signal) => apiFetch(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}${query}`, { method: 'DELETE', signal }),
      PROJECT_REQUEST_TIMEOUT_MS,
      signal
    );
  } catch (error) {
    if (!(error instanceof RequestDeadlineError)) {
      throw error;
    }
    // Deletion is idempotent: a settled repeat returns 204 even when the first request already removed the project.
    await withRequestDeadline(
      (signal) => apiFetch(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}${query}`, { method: 'DELETE', signal }),
      PROJECT_REQUEST_TIMEOUT_MS,
      signal
    );
  }
};

export const getProjectBoardSnapshot = (projectId: string, signal?: AbortSignal): Promise<ProjectBoardSnapshotDTO> =>
  apiFetchJson<ProjectBoardSnapshotDTO>(`${PROJECTS_BASE}/${encodeURIComponent(projectId)}/board-snapshot`, {
    signal,
  });

export const isProjectConflictError = (error: unknown): boolean => error instanceof ApiError && error.status === 409;

export const isProjectNotFoundError = (error: unknown): boolean => error instanceof ApiError && error.status === 404;

export interface ProjectCanvasSchemaCompatibilityRefusal {
  minimumCanvasSchemaVersion: number;
  maxCanvasSchemaVersion: number;
}

export const getProjectCanvasSchemaCompatibilityRefusal = (
  error: unknown
): ProjectCanvasSchemaCompatibilityRefusal | null => {
  if (!(error instanceof ApiError) || error.status !== 412) {
    return null;
  }

  try {
    const body = JSON.parse(error.message) as {
      detail?: {
        code?: unknown;
        max_canvas_schema_version?: unknown;
        minimum_canvas_schema_version?: unknown;
      };
    };
    const code = body.detail?.code;
    const minimum = body.detail?.minimum_canvas_schema_version;
    const maximum = body.detail?.max_canvas_schema_version;

    return code === 'canvas_schema_unsupported' &&
      typeof minimum === 'number' &&
      Number.isInteger(minimum) &&
      minimum >= 1 &&
      typeof maximum === 'number' &&
      Number.isInteger(maximum) &&
      maximum >= 1 &&
      minimum > maximum
      ? { maxCanvasSchemaVersion: maximum, minimumCanvasSchemaVersion: minimum }
      : null;
  } catch {
    return null;
  }
};

export interface ProjectDocumentSchemaRefusal {
  minimumDocumentSchemaVersion: number;
  maxDocumentSchemaVersion: number;
}

export const getProjectDocumentSchemaRefusal = (error: unknown): ProjectDocumentSchemaRefusal | null => {
  if (!(error instanceof ApiError) || error.status !== 412) {
    return null;
  }
  try {
    const detail = (JSON.parse(error.message) as { detail?: Record<string, unknown> }).detail;
    const minimum = detail?.minimum_document_schema_version;
    const maximum = detail?.max_document_schema_version;
    return detail?.code === 'document_schema_unsupported' &&
      Number.isSafeInteger(minimum) &&
      Number.isSafeInteger(maximum) &&
      (minimum as number) > (maximum as number)
      ? { maxDocumentSchemaVersion: maximum as number, minimumDocumentSchemaVersion: minimum as number }
      : null;
  } catch {
    return null;
  }
};

export const isProjectCanvasSchemaUnsupportedError = (error: unknown): boolean =>
  getProjectCanvasSchemaCompatibilityRefusal(error) !== null;

export interface ProjectWriteSizeRefusal {
  actualBytes: number;
  kind: 'document' | 'request';
  maxBytes: number;
}

export const getProjectWriteSizeRefusal = (error: unknown): ProjectWriteSizeRefusal | null => {
  if (!(error instanceof ApiError) || error.status !== 413) {
    return null;
  }

  try {
    const body = JSON.parse(error.message) as {
      detail?: { actual_bytes?: unknown; code?: unknown; max_bytes?: unknown };
    };
    const actualBytes = body.detail?.actual_bytes;
    const maxBytes = body.detail?.max_bytes;

    const kind =
      body.detail?.code === 'project_document_too_large'
        ? 'document'
        : body.detail?.code === 'project_request_too_large'
          ? 'request'
          : null;

    return kind !== null &&
      typeof actualBytes === 'number' &&
      Number.isSafeInteger(actualBytes) &&
      typeof maxBytes === 'number' &&
      Number.isSafeInteger(maxBytes) &&
      maxBytes > 0 &&
      actualBytes > maxBytes
      ? { actualBytes, kind, maxBytes }
      : null;
  } catch {
    return null;
  }
};

/** Only confirmed project absence permits compensation. */
export class ProjectCreateAbsentError extends Error {
  readonly cause: unknown;

  constructor(cause: unknown) {
    super('The project was not created.');
    this.name = 'ProjectCreateAbsentError';
    this.cause = cause;
  }
}

/**
 * Settle ambiguous creates with a second POST (GET 404 can race the initial commit). Transport failure stays unknown
 * and must never authorize deletion.
 */
const settleProjectCreate = async (
  request: ProjectCreateRequest,
  owner: AccountScope
): Promise<ProjectWithWorkspaceDTO> => {
  const projectId = request.project_id;
  const body = serializeCreateProjectRequest(request);
  const writeBudget = createProjectWriteRetryBudget();

  const settleConflict = async (conflict: unknown): Promise<ProjectWithWorkspaceDTO> => {
    if (projectId === undefined) {
      throw conflict;
    }

    try {
      const record = await getProject(projectId, owner.signal);

      if (!projectRecordMatchesCreate(record, body)) {
        throw conflict;
      }

      return record;
    } catch (readError) {
      assertAccountScopeCurrent(owner);

      throw isProjectNotFoundError(readError) ? new ProjectCreateAbsentError(conflict) : conflict;
    }
  };

  /** A 404 here may predate the commit, so proves nothing. */
  const settleUnanswered = async (unanswered: unknown, id: string): Promise<ProjectWithWorkspaceDTO> => {
    let record: ProjectWithWorkspaceDTO | null = null;

    try {
      record = await getProject(id, owner.signal);
    } catch {
      // Still unknown.
    }
    assertAccountScopeCurrent(owner);

    if (record === null || !projectRecordMatchesCreate(record, body)) {
      throw unanswered;
    }

    return record;
  };

  const classify = (error: unknown): Promise<ProjectWithWorkspaceDTO> => {
    if (isProjectConflictError(error)) {
      return settleConflict(error);
    }

    throw isRefusal(error) ? new ProjectCreateAbsentError(error) : error;
  };

  try {
    return await createProjectFromBody(body, owner.signal, writeBudget);
  } catch (error) {
    assertAccountScopeCurrent(owner);

    // The caller must choose the ID so retries are idempotent.
    if (projectId === undefined || !isIndeterminate(error)) {
      return classify(error);
    }

    try {
      return await createProjectFromBody(body, owner.signal, writeBudget);
    } catch (retryError) {
      assertAccountScopeCurrent(owner);

      // An ambiguous retry must retain uploads; unknown never authorizes cleanup.
      if (isIndeterminate(retryError) || isProjectWriteBusyError(retryError)) {
        return settleUnanswered(error, projectId);
      }

      return classify(retryError);
    }
  }
};

export const createProjectSettled = async (
  request: ProjectCreateRequest,
  owner: AccountScope
): Promise<ProjectWithWorkspaceDTO> => {
  const record = await settleProjectCreate(request, owner);

  // The server mints the project's board with it, so board lists fetched before now are missing it.
  invalidateBoardLists(owner);

  return record;
};

/** Loaded lazily to keep the gallery data layer off the launchpad's startup graph. */
export const invalidateBoardLists = (owner: AccountScope): void => {
  void import('@features/gallery/queries').then(({ galleryKeys }) =>
    queryClient.invalidateQueries({ queryKey: galleryKeys.boardsForAccount(owner) })
  );
};

const isRefusal = (error: unknown): boolean => error instanceof ApiError && error.status >= 400 && error.status < 500;

const isIndeterminate = (error: unknown): boolean => !(error instanceof ApiError) || error.status >= 500;

export const getClientStateValue = (key: string, signal?: AbortSignal): Promise<string | null> =>
  withRequestDeadline(
    (signal) =>
      apiFetchJson<string | null>(`${CLIENT_STATE_BASE}/get_by_key?key=${encodeURIComponent(key)}`, { signal }),
    CLIENT_STATE_READ_TIMEOUT_MS,
    signal
  );

/** The endpoint takes a JSON-encoded string body, hence the stringify of a string. */
export const setClientStateValue = async (key: string, value: string, signal?: AbortSignal): Promise<void> => {
  // Must settle before callers release ordering queues; a timeout cannot fence a late commit.
  await apiFetchJson<string>(`${CLIENT_STATE_BASE}/set_by_key?key=${encodeURIComponent(key)}`, {
    body: JSON.stringify(value),
    method: 'POST',
    signal,
  });
};

export const deleteClientStateValue = async (key: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`${CLIENT_STATE_BASE}/delete_by_key?key=${encodeURIComponent(key)}`, { method: 'POST', signal });
};
