import type {
  IntermediatesAffectedDocument,
  IntermediatesAffectedDocumentKind,
  IntermediatesCleanupMode,
  IntermediatesImpact,
  IntermediatesKindCounts,
  IntermediatesOperation,
  IntermediatesOperationProgress,
  IntermediatesPreview,
  IntermediatesRow,
  IntermediatesScope,
  IntermediatesScopeTarget,
  IntermediatesSummary,
} from '@features/intermediates/core/types';

import { apiFetchJson } from '@platform/transport/http';

import type { IntermediatesSummaryParams } from './keys';

const BASE = '/api/v1/intermediates';

// Wire DTOs are private; every consumer reads the camelCase read models.

interface KindCountsDTO {
  safe: number;
  referenced: number;
  active: number;
  recent: number;
}

interface ScopeTargetDTO {
  user_id: string;
  project_id: string | null;
}

/** One flat model on the wire; each kind reads the fields it needs and the rest stay empty. */
interface ScopeDTO {
  kind: 'selection' | 'owner' | 'everyone' | 'matching';
  targets?: ScopeTargetDTO[];
  user_id?: string | null;
  project_id?: string | null;
  search?: string | null;
  excluded?: ScopeTargetDTO[];
}

interface RowDTO extends ScopeTargetDTO {
  user_display_name: string | null;
  user_email: string | null;
  project_name: string | null;
  cover_image_name: string | null;
  images: KindCountsDTO;
  videos: KindCountsDTO;
  reclaimable_bytes: number;
  referenced_bytes: number;
  unknown_size_count: number;
}

interface SummaryDTO {
  items: RowDTO[];
  total: number;
  offset: number;
  limit: number;
  totals: {
    rows: number;
    safe_images: number;
    safe_videos: number;
    in_use_images: number;
    in_use_videos: number;
    reclaimable_bytes: number;
    unknown_size_count: number;
  };
  recent_grace_seconds: number;
  measuring: boolean;
  can_manage_everyone: boolean;
}

interface ImpactDTO {
  delete_images: number;
  delete_videos: number;
  keep_referenced_images: number;
  keep_referenced_videos: number;
  keep_active_images: number;
  keep_active_videos: number;
  keep_recent_images: number;
  keep_recent_videos: number;
  reclaimable_bytes: number;
  unknown_size_count: number;
}

interface AffectedDocumentDTO {
  kind: IntermediatesAffectedDocumentKind;
  user_id: string;
  user_display_name: string | null;
  user_email: string | null;
  owner_id: string;
  name: string | null;
  references: number;
}

interface PreviewDTO {
  preview_id: string;
  mode: IntermediatesCleanupMode;
  scope: ScopeDTO;
  created_at: string;
  expires_at: string;
  target_rows: number;
  impact: ImpactDTO;
  affected_documents: AffectedDocumentDTO[];
  affected_documents_total: number;
}

interface ProgressDTO {
  processed_images: number;
  processed_videos: number;
  deleted_images: number;
  deleted_videos: number;
  retained_images: number;
  retained_videos: number;
  failed_images: number;
  failed_videos: number;
  reclaimed_bytes: number;
  unknown_size_count: number;
  pending_disk_cleanup: number;
}

export interface IntermediatesOperationDTO {
  operation_id: string;
  user_id: string;
  mode: IntermediatesCleanupMode;
  scope: ScopeDTO;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  target_images: number;
  target_videos: number;
  progress: ProgressDTO;
}

const mapCounts = (dto: KindCountsDTO): IntermediatesKindCounts => ({
  active: dto.active,
  recent: dto.recent,
  referenced: dto.referenced,
  safe: dto.safe,
});

const mapTarget = (dto: ScopeTargetDTO): IntermediatesScopeTarget => ({
  projectId: dto.project_id,
  userId: dto.user_id,
});

const toTargetDTO = (target: IntermediatesScopeTarget): ScopeTargetDTO => ({
  project_id: target.projectId,
  user_id: target.userId,
});

const mapRow = (dto: RowDTO): IntermediatesRow => ({
  coverImageName: dto.cover_image_name,
  images: mapCounts(dto.images),
  projectId: dto.project_id,
  projectName: dto.project_name,
  reclaimableBytes: dto.reclaimable_bytes,
  referencedBytes: dto.referenced_bytes,
  unknownSizeCount: dto.unknown_size_count,
  userDisplayName: dto.user_display_name,
  userEmail: dto.user_email,
  userId: dto.user_id,
  videos: mapCounts(dto.videos),
});

const mapScope = (dto: ScopeDTO): IntermediatesScope => {
  if (dto.kind === 'everyone') {
    return { kind: 'everyone' };
  }
  if (dto.kind === 'owner') {
    return { kind: 'owner', userId: dto.user_id ?? '' };
  }
  if (dto.kind === 'matching') {
    return {
      excluded: (dto.excluded ?? []).map(mapTarget),
      kind: 'matching',
      projectId: dto.project_id ?? null,
      search: dto.search ?? null,
      userId: dto.user_id ?? null,
    };
  }

  return { kind: 'selection', targets: (dto.targets ?? []).map(mapTarget) };
};

const toScopeDTO = (scope: IntermediatesScope): ScopeDTO => {
  const empty: Required<ScopeDTO> = {
    excluded: [],
    kind: scope.kind,
    project_id: null,
    search: null,
    targets: [],
    user_id: null,
  };
  if (scope.kind === 'owner') {
    return { ...empty, user_id: scope.userId };
  }
  if (scope.kind === 'selection') {
    return { ...empty, targets: scope.targets.map(toTargetDTO) };
  }
  if (scope.kind === 'matching') {
    return {
      ...empty,
      excluded: scope.excluded.map(toTargetDTO),
      project_id: scope.projectId,
      search: scope.search,
      user_id: scope.userId,
    };
  }

  return empty;
};

const mapImpact = (dto: ImpactDTO): IntermediatesImpact => ({
  deleteImages: dto.delete_images,
  deleteVideos: dto.delete_videos,
  keepActiveImages: dto.keep_active_images,
  keepActiveVideos: dto.keep_active_videos,
  keepRecentImages: dto.keep_recent_images,
  keepRecentVideos: dto.keep_recent_videos,
  keepReferencedImages: dto.keep_referenced_images,
  keepReferencedVideos: dto.keep_referenced_videos,
  reclaimableBytes: dto.reclaimable_bytes,
  unknownSizeCount: dto.unknown_size_count,
});

const mapAffectedDocument = (dto: AffectedDocumentDTO): IntermediatesAffectedDocument => ({
  kind: dto.kind,
  name: dto.name,
  ownerId: dto.owner_id,
  references: dto.references,
  userDisplayName: dto.user_display_name,
  userEmail: dto.user_email,
  userId: dto.user_id,
});

const mapProgress = (dto: ProgressDTO): IntermediatesOperationProgress => ({
  deletedImages: dto.deleted_images,
  deletedVideos: dto.deleted_videos,
  failedImages: dto.failed_images,
  failedVideos: dto.failed_videos,
  pendingDiskCleanup: dto.pending_disk_cleanup,
  processedImages: dto.processed_images,
  processedVideos: dto.processed_videos,
  reclaimedBytes: dto.reclaimed_bytes,
  retainedImages: dto.retained_images,
  retainedVideos: dto.retained_videos,
  unknownSizeCount: dto.unknown_size_count,
});

export const mapIntermediatesOperation = (dto: IntermediatesOperationDTO): IntermediatesOperation => ({
  completedAt: dto.completed_at,
  createdAt: dto.created_at,
  error: dto.error,
  mode: dto.mode,
  operationId: dto.operation_id,
  progress: mapProgress(dto.progress),
  scope: mapScope(dto.scope),
  startedAt: dto.started_at,
  status: dto.status,
  targetImages: dto.target_images,
  targetVideos: dto.target_videos,
  userId: dto.user_id,
});

const PROGRESS_KEYS = [
  'processed_images',
  'processed_videos',
  'deleted_images',
  'deleted_videos',
  'retained_images',
  'retained_videos',
  'failed_images',
  'failed_videos',
  'reclaimed_bytes',
  'unknown_size_count',
  'pending_disk_cleanup',
] as const satisfies readonly (keyof ProgressDTO)[];
const OPERATION_STATUSES = new Set<unknown>(['pending', 'running', 'completed', 'failed']);
const CLEANUP_MODES = new Set<unknown>(['safe', 'force']);

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;
const isCount = (value: unknown): boolean => typeof value === 'number' && Number.isFinite(value);
const isNullableString = (value: unknown): boolean => value === null || typeof value === 'string';
const isTargetList = (value: unknown): boolean =>
  value === undefined ||
  (Array.isArray(value) &&
    value.every(
      (target) => isRecord(target) && typeof target.user_id === 'string' && isNullableString(target.project_id)
    ));

const isScopeDTO = (value: unknown): value is ScopeDTO =>
  isRecord(value) &&
  (value.kind === 'everyone' ||
    (value.kind === 'owner' && isNullableString(value.user_id)) ||
    (value.kind === 'selection' && isTargetList(value.targets)) ||
    (value.kind === 'matching' &&
      [value.user_id, value.project_id, value.search].every(
        (field) => field === undefined || isNullableString(field)
      ) &&
      isTargetList(value.excluded)));

const isOperationDTO = (value: unknown): value is IntermediatesOperationDTO =>
  isRecord(value) &&
  typeof value.operation_id === 'string' &&
  typeof value.user_id === 'string' &&
  CLEANUP_MODES.has(value.mode) &&
  OPERATION_STATUSES.has(value.status) &&
  isScopeDTO(value.scope) &&
  isCount(value.target_images) &&
  isCount(value.target_videos) &&
  isRecord(value.progress) &&
  PROGRESS_KEYS.every((key) => isCount((value.progress as Record<string, unknown>)[key])) &&
  [value.created_at, value.started_at, value.completed_at, value.error].every(isNullableString);

/** Reads a socket event's operation, which arrives unvalidated; a malformed event yields null. */
export const parseIntermediatesOperationEvent = (payload: unknown): IntermediatesOperation | null =>
  isRecord(payload) && isOperationDTO(payload.operation) ? mapIntermediatesOperation(payload.operation) : null;

const buildSummaryUrl = (params: IntermediatesSummaryParams): string => {
  const query = new URLSearchParams();

  if (params.ownerId) {
    query.set('owner_id', params.ownerId);
  }
  if (params.projectId) {
    query.set('project_id', params.projectId);
  }
  if (params.search?.trim()) {
    query.set('search', params.search.trim());
  }
  if (params.sort) {
    query.set('sort', params.sort);
  }
  if (params.order) {
    query.set('order', params.order);
  }
  if (params.offset !== undefined && params.offset > 0) {
    query.set('offset', String(params.offset));
  }
  if (params.limit !== undefined) {
    query.set('limit', String(params.limit));
  }

  const encoded = query.toString();

  return encoded ? `${BASE}/summary?${encoded}` : `${BASE}/summary`;
};

export const getIntermediatesSummary = async (
  params: IntermediatesSummaryParams,
  signal?: AbortSignal
): Promise<IntermediatesSummary> => {
  const dto = await apiFetchJson<SummaryDTO>(buildSummaryUrl(params), { signal });

  return {
    canManageEveryone: dto.can_manage_everyone,
    items: dto.items.map(mapRow),
    limit: dto.limit,
    measuring: dto.measuring,
    offset: dto.offset,
    recentGraceSeconds: dto.recent_grace_seconds,
    total: dto.total,
    totals: {
      inUseImages: dto.totals.in_use_images,
      inUseVideos: dto.totals.in_use_videos,
      reclaimableBytes: dto.totals.reclaimable_bytes,
      rows: dto.totals.rows,
      safeImages: dto.totals.safe_images,
      safeVideos: dto.totals.safe_videos,
      unknownSizeCount: dto.totals.unknown_size_count,
    },
  };
};

export const createIntermediatesPreview = async (
  request: { mode: IntermediatesCleanupMode; scope: IntermediatesScope },
  signal?: AbortSignal
): Promise<IntermediatesPreview> => {
  const dto = await apiFetchJson<PreviewDTO>(`${BASE}/previews`, {
    body: JSON.stringify({ mode: request.mode, scope: toScopeDTO(request.scope) }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
    signal,
  });

  return {
    affectedDocuments: dto.affected_documents.map(mapAffectedDocument),
    affectedDocumentsTotal: dto.affected_documents_total,
    createdAt: dto.created_at,
    expiresAt: dto.expires_at,
    impact: mapImpact(dto.impact),
    mode: dto.mode,
    previewId: dto.preview_id,
    scope: mapScope(dto.scope),
    targetRows: dto.target_rows,
  };
};

/** Long enough for a slow server to accept; short enough that a hung request cannot lock the confirmation. */
const START_TIMEOUT_MS = 30_000;

/**
 * Confirms a preview, which the server accepts once. Rejects with a `TimeoutError` DOMException when the server does
 * not answer in time; the run may still have started, and `listIntermediatesOperations` finds it.
 */
export const startIntermediatesOperation = async (
  request: { previewId: string },
  signal: AbortSignal
): Promise<IntermediatesOperation> =>
  mapIntermediatesOperation(
    await apiFetchJson<IntermediatesOperationDTO>(`${BASE}/operations`, {
      body: JSON.stringify({ preview_id: request.previewId }),
      headers: { 'Content-Type': 'application/json' },
      method: 'POST',
      signal: AbortSignal.any([signal, AbortSignal.timeout(START_TIMEOUT_MS)]),
    })
  );

/** The account's running and recently settled operations, newest first. */
export const listIntermediatesOperations = async (signal?: AbortSignal): Promise<IntermediatesOperation[]> =>
  (await apiFetchJson<{ items: IntermediatesOperationDTO[] }>(`${BASE}/operations`, { signal })).items.map(
    mapIntermediatesOperation
  );

export const getIntermediatesOperation = async (
  operationId: string,
  signal?: AbortSignal
): Promise<IntermediatesOperation> =>
  mapIntermediatesOperation(
    await apiFetchJson<IntermediatesOperationDTO>(`${BASE}/operations/${encodeURIComponent(operationId)}`, {
      signal,
    })
  );
