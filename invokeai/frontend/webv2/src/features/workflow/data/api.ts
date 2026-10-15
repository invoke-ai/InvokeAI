import { ApiError, apiFetch, apiFetchJson } from '@platform/transport/http';

/** Round-trip library WorkflowV3 payloads through workflowJson when crossing the document boundary. */

export type WorkflowLibraryCategory = 'user' | 'default';

export interface WorkflowLibraryListItem {
  workflow_id: string;
  name: string;
  description: string;
  category: WorkflowLibraryCategory;
  /** Monotonic content revision; every write of the template's content increments it. */
  revision: number;
  user_id?: string;
  is_public?: boolean;
  call_saved_workflow_compatibility?: WorkflowCallCompatibility | null;
  tags?: string | null;
  created_at?: string;
  updated_at?: string;
  opened_at?: string | null;
  last_run_at?: string | null;
  thumbnail_url?: string | null;
}

export interface WorkflowCallCompatibility {
  is_callable: boolean;
  reason: string;
  message: string | null;
}

export interface WorkflowLibraryPage {
  items: WorkflowLibraryListItem[];
  page: number;
  pages: number;
  total: number;
}

export interface ListWorkflowsParams {
  category?: WorkflowLibraryCategory;
  categories?: WorkflowLibraryCategory[];
  page: number;
  perPage?: number;
  query?: string;
  tags?: string[];
  isPublic?: boolean;
  callable?: boolean;
  orderBy?: 'updated_at' | 'name';
  direction?: 'ASC' | 'DESC';
  signal?: AbortSignal;
}

export const listLibraryWorkflows = ({
  category,
  categories,
  callable,
  direction = 'DESC',
  page,
  perPage = 20,
  query,
  tags,
  isPublic,
  orderBy = 'updated_at',
  signal,
}: ListWorkflowsParams): Promise<WorkflowLibraryPage> => {
  const params = new URLSearchParams({
    direction,
    order_by: orderBy,
    page: String(page),
    per_page: String(perPage),
  });

  for (const workflowCategory of categories ?? (category ? [category] : [])) {
    params.append('categories', workflowCategory);
  }

  if (isPublic !== undefined) {
    params.set('is_public', String(isPublic));
  }

  if (callable !== undefined) {
    params.set('callable', String(callable));
  }

  if (query?.trim()) {
    params.set('query', query.trim());
  }

  for (const tag of tags ?? []) {
    params.append('tags', tag);
  }

  return apiFetchJson<WorkflowLibraryPage>(`/api/v1/workflows/?${params.toString()}`, { signal });
};

export interface WorkflowRecordDTO extends WorkflowLibraryListItem {
  workflow_id: string;
  name: string;
  workflow: Record<string, unknown>;
}

export const getLibraryWorkflowRecord = (workflowId: string, signal?: AbortSignal): Promise<WorkflowRecordDTO> =>
  apiFetchJson<WorkflowRecordDTO>(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, { signal });

/** A write the server refused for a structured reason; other failures stay `ApiError`s. */
export class WorkflowLibraryWriteRefusedError extends Error {
  readonly reason: 'revision-conflict' | 'id-conflict' | 'bundled' | 'forbidden' | 'missing' | 'invalid';
  /** The template's revision at refusal time, when the server reported one. */
  readonly currentRevision: number | null;

  constructor(
    reason: WorkflowLibraryWriteRefusedError['reason'],
    message: string,
    currentRevision: number | null = null
  ) {
    super(message);
    this.name = 'WorkflowLibraryWriteRefusedError';
    this.reason = reason;
    this.currentRevision = currentRevision;
  }
}

const parseErrorDetail = (error: ApiError): Record<string, unknown> | string | null => {
  try {
    const parsed = JSON.parse(error.message) as { detail?: unknown };

    return typeof parsed.detail === 'string' || (typeof parsed.detail === 'object' && parsed.detail !== null)
      ? (parsed.detail as Record<string, unknown> | string)
      : null;
  } catch {
    return null;
  }
};

/** Translate the workflow endpoints' 400/403/404/409 answers; anything else is not a refusal the caller can act on. */
export const toWorkflowLibraryWriteRefusal = (error: unknown): WorkflowLibraryWriteRefusedError | null => {
  if (error instanceof WorkflowLibraryWriteRefusedError) {
    return error;
  }

  if (!(error instanceof ApiError)) {
    return null;
  }

  const detail = parseErrorDetail(error);
  const message = typeof detail === 'string' ? detail : typeof detail?.message === 'string' ? detail.message : '';

  if (error.status === 409 && typeof detail === 'object' && detail?.reason === 'revision-conflict') {
    const current = detail.current_revision;

    return new WorkflowLibraryWriteRefusedError(
      'revision-conflict',
      message,
      typeof current === 'number' ? current : null
    );
  }

  if (error.status === 409) {
    return new WorkflowLibraryWriteRefusedError('id-conflict', message);
  }

  if (error.status === 403) {
    return new WorkflowLibraryWriteRefusedError(/bundled/i.test(message) ? 'bundled' : 'forbidden', message);
  }

  if (error.status === 404) {
    return new WorkflowLibraryWriteRefusedError('missing', message);
  }

  if (error.status === 400 || error.status === 422) {
    return new WorkflowLibraryWriteRefusedError('invalid', message);
  }

  return null;
};

const rethrowAsRefusal = (error: unknown): never => {
  throw toWorkflowLibraryWriteRefusal(error) ?? error;
};

export interface CreateLibraryWorkflowOptions {
  /** A client-reserved UUID; resending it after a lost response returns the record the first send created. */
  reservedId?: string;
  signal?: AbortSignal;
}

export const createLibraryWorkflowRecord = async (
  workflow: Record<string, unknown>,
  { reservedId, signal }: CreateLibraryWorkflowOptions = {}
): Promise<WorkflowRecordDTO> => {
  const { id: _id, ...workflowWithoutId } = workflow;

  try {
    return await apiFetchJson<WorkflowRecordDTO>('/api/v1/workflows/', {
      body: JSON.stringify({ workflow: workflowWithoutId, ...(reservedId ? { workflow_id: reservedId } : {}) }),
      method: 'POST',
      signal,
    });
  } catch (error) {
    return rethrowAsRefusal(error);
  }
};

export const createLibraryWorkflow = async (workflow: Record<string, unknown>, signal?: AbortSignal): Promise<string> =>
  (await createLibraryWorkflowRecord(workflow, { signal })).workflow_id;

export interface UpdateLibraryWorkflowOptions {
  /** The revision the caller last observed; the server refuses the write once the template moved past it. */
  expectedRevision?: number;
  signal?: AbortSignal;
}

export const updateLibraryWorkflow = async (
  workflowId: string,
  workflow: Record<string, unknown>,
  { expectedRevision, signal }: UpdateLibraryWorkflowOptions = {}
): Promise<WorkflowRecordDTO> => {
  try {
    return await apiFetchJson<WorkflowRecordDTO>(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, {
      body: JSON.stringify({
        workflow: { ...workflow, id: workflowId },
        ...(expectedRevision === undefined ? {} : { expected_revision: expectedRevision }),
      }),
      method: 'PATCH',
      signal,
    });
  } catch (error) {
    return rethrowAsRefusal(error);
  }
};

export const deleteLibraryWorkflow = async (workflowId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}`, { method: 'DELETE', signal });
};

/**
 * The server stores a 256px copy and serves it from a fixed path; every list/get response appends a fresh query to
 * `thumbnail_url`, so invalidating the library cache is what makes a replaced image load.
 */
export const setLibraryWorkflowThumbnail = async (
  workflowId: string,
  image: Blob,
  signal?: AbortSignal
): Promise<void> => {
  const body = new FormData();

  body.append('image', image);
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}/thumbnail`, { body, method: 'PUT', signal });
};

export const deleteLibraryWorkflowThumbnail = async (workflowId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}/thumbnail`, { method: 'DELETE', signal });
};

export const touchLibraryWorkflowOpenedAt = async (workflowId: string, signal?: AbortSignal): Promise<void> => {
  await apiFetch(`/api/v1/workflows/i/${encodeURIComponent(workflowId)}/opened_at`, { method: 'PUT', signal });
};

export interface GetWorkflowTagCountsParams {
  categories?: WorkflowLibraryCategory[];
  hasBeenOpened?: boolean;
  isPublic?: boolean;
  tags: string[];
  signal?: AbortSignal;
}

/** Thin wrapper over `GET /api/v1/workflows/counts_by_tag`; returns the route's JSON verbatim. */
export const getWorkflowTagCounts = ({
  categories,
  hasBeenOpened,
  isPublic,
  tags,
  signal,
}: GetWorkflowTagCountsParams): Promise<Record<string, number>> => {
  const params = new URLSearchParams();

  for (const tag of tags) {
    params.append('tags', tag);
  }

  for (const category of categories ?? []) {
    params.append('categories', category);
  }

  if (hasBeenOpened !== undefined) {
    params.set('has_been_opened', String(hasBeenOpened));
  }

  if (isPublic !== undefined) {
    params.set('is_public', String(isPublic));
  }

  return apiFetchJson<Record<string, number>>(`/api/v1/workflows/counts_by_tag?${params.toString()}`, { signal });
};

export interface GetAllWorkflowTagsParams {
  categories?: WorkflowLibraryCategory[];
  isPublic?: boolean;
  signal?: AbortSignal;
}

/** Thin wrapper over `GET /api/v1/workflows/tags`; returns the route's JSON verbatim. */
export const getAllWorkflowTags = ({ categories, isPublic, signal }: GetAllWorkflowTagsParams = {}): Promise<
  string[]
> => {
  const params = new URLSearchParams();

  for (const category of categories ?? []) {
    params.append('categories', category);
  }

  if (isPublic !== undefined) {
    params.set('is_public', String(isPublic));
  }

  const query = params.toString();

  return apiFetchJson<string[]>(`/api/v1/workflows/tags${query ? `?${query}` : ''}`, { signal });
};
