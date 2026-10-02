import type { ProjectSchemaRefusal } from './projectFlush';

export type ProjectDraftConflict = { kind: 'deleted' } | { kind: 'revision'; serverRevision: number };

export interface ProjectDraftInput {
  baseMinimumCanvasSchemaVersion?: number;
  baseRevision: number | null;
  documentJson: string;
  documentSchemaVersion: number;
  editorSessionId: string;
  generation: number;
  projectId: string;
  updatedAt: number;
  writerToken: string;
}

interface ProjectDraftBase extends ProjectDraftInput {
  copyDocumentByteSize?: number;
  copyDocumentJson?: string;
  copyProjectId?: string;
  copyProjectGeneration?: number;
  copyProjectMinimumCanvasSchemaVersion?: number;
  copyProjectName?: string;
  copySourceProjectName?: string;
  documentByteSize: number;
}

export type ProjectDraft =
  | (ProjectDraftBase & { state: 'dirty' })
  | (ProjectDraftBase & { conflict: ProjectDraftConflict; state: 'conflict' })
  | (ProjectDraftBase & { refusal: ProjectSchemaRefusal; state: 'schema-refused' });

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

export type ProjectDraftMetadata = DistributiveOmit<ProjectDraft, 'copyDocumentJson' | 'documentJson'> & {
  metadataRevision: number;
};

export interface ProjectDraftBody {
  copyDocumentJson?: string;
  documentByteSize: number;
  documentJson: string;
  editorSessionId: string;
  generation: number;
  projectId: string;
  recordType: 'draft-body';
}

export interface ProjectDraftSummary {
  documentByteSize: number | null;
  documentSchemaVersion: number | null;
  editorSessionId: string;
  generation: number | null;
  projectId: string;
  state: ProjectDraft['state'] | 'corrupt';
  updatedAt: number | null;
}

export type ProjectDraftKey = [projectId: string, editorSessionId: string];

export type ProjectDraftWriterClaim =
  | {
      adoptedByEditorSessionId?: string;
      editorSessionId: string;
      fenceReason: 'corrupt-cleanup' | 'moved';
      metadataRevision: number;
      projectId: string;
      retargetedToProjectId?: string;
      retargetedToRevision?: number;
      state: 'fenced';
      updatedAt: number;
      writerToken: string;
    }
  | {
      editorSessionId: string;
      metadataRevision: number;
      projectId: string;
      state: 'active';
      updatedAt: number;
      writerToken: string;
    };

export type ProjectDraftPageResult =
  | { items: ProjectDraftSummary[]; kind: 'available'; nextCursor: ProjectDraftKey | null }
  | { kind: 'unavailable' };
export type ProjectDraftListResult =
  | { items: ProjectDraftSummary[]; kind: 'available'; nextCursor: string | null }
  | { kind: 'unavailable' };
export type ProjectDraftStageResult = {
  kind:
    | 'corrupt'
    | 'fenced'
    | 'generation-conflict'
    | 'quota'
    | 'replayed'
    | 'stale'
    | 'stored'
    | 'too-large'
    | 'unavailable';
};
export type ProjectDraftClaimResult = { kind: 'claimed' | 'corrupt' | 'fenced' | 'missing' | 'quota' | 'unavailable' };
export type ProjectDraftStartWriterResult = {
  kind: 'corrupt' | 'fenced' | 'occupied' | 'quota' | 'started' | 'unavailable';
};
export type ProjectDraftAdoptionResult = {
  kind: 'adopted' | 'corrupt' | 'missing' | 'occupied' | 'quota' | 'unavailable';
};
export type ProjectDraftDeleteResult = { kind: 'corrupt' | 'deleted' | 'fenced' | 'unavailable' };
export interface ProjectDraftRetargetHandoff {
  editorSessionId: string;
  projectId: string;
  revision: number;
  targetProjectId: string;
  updatedAt: number;
}
export type ProjectDraftRetargetCursor = [projectId: string, editorSessionId: string, targetProjectId: string];
export type ProjectDraftRetargetListResult =
  | { items: ProjectDraftRetargetHandoff[]; kind: 'available'; nextCursor: ProjectDraftRetargetCursor | null }
  | { kind: 'unavailable' };
export type ProjectDraftRetargetAcknowledgeResult = { kind: 'deleted' | 'stale' | 'unavailable' };
export type ProjectDraftCorruptDeleteResult = { kind: 'deleted' | 'not-corrupt' | 'unavailable' };
export type ProjectDraftGetResult =
  | { draft: ProjectDraft; kind: 'found' }
  | { kind: 'empty'; writerState: ProjectDraftWriterClaim['state']; writerToken: string }
  | { kind: 'retargeted'; projectId: string; revision: number; writerToken: string }
  | { kind: 'corrupt' | 'missing' | 'unavailable' };
export type ProjectDraftSettlementResult =
  | { draft: ProjectDraft; kind: 'marked' | 'rebased' }
  | { draft: ProjectDraft | null; kind: 'retargeted' }
  | {
      kind: 'corrupt' | 'deleted' | 'fenced' | 'missing' | 'occupied' | 'quota' | 'stale' | 'too-large' | 'unavailable';
    };
export interface ProjectDraftCopyReservation {
  copyDocumentByteSize: number;
  copyDocumentJson: string;
  copyProjectGeneration: number;
  copyProjectId: string;
  copyProjectMinimumCanvasSchemaVersion: number;
  copyProjectName: string;
  copySourceProjectName: string;
}
export type ProjectDraftCopyReservationResult =
  | (ProjectDraftCopyReservation & { kind: 'reserved' })
  | { kind: 'corrupt' | 'fenced' | 'missing' | 'quota' | 'stale' | 'unavailable' };

export interface RetargetAcknowledgedCopyOptions {
  acknowledgedRevision: number;
  copyProjectId: string;
  editorSessionId: string;
  projectId: string;
  retargetDocument(documentJson: string): string;
  sentGeneration: number;
  writerToken: string;
}

export interface ProjectDraftStore {
  readonly availability: 'available' | 'unavailable';
  adopt(
    projectId: string,
    fromEditorSessionId: string,
    toEditorSessionId: string,
    toWriterToken: string
  ): Promise<ProjectDraftAdoptionResult>;
  acknowledgeRetarget(
    projectId: string,
    editorSessionId: string,
    targetProjectId: string
  ): Promise<ProjectDraftRetargetAcknowledgeResult>;
  claimWriter(
    projectId: string,
    editorSessionId: string,
    expectedWriterToken: string,
    nextWriterToken: string
  ): Promise<ProjectDraftClaimResult>;
  close(): void;
  delete(projectId: string, editorSessionId: string, writerToken: string): Promise<ProjectDraftDeleteResult>;
  deleteCorrupt(projectId: string, editorSessionId: string): Promise<ProjectDraftCorruptDeleteResult>;
  get(projectId: string, editorSessionId: string): Promise<ProjectDraftGetResult>;
  list(options?: { after?: ProjectDraftKey; limit?: number }): Promise<ProjectDraftPageResult>;
  listForProject(projectId: string, options?: { after?: string; limit?: number }): Promise<ProjectDraftListResult>;
  listRetargets(options?: {
    after?: ProjectDraftRetargetCursor;
    limit?: number;
  }): Promise<ProjectDraftRetargetListResult>;
  reserveCopyIdentity(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    proposed: ProjectDraftCopyReservation,
    replaceCopyProjectId?: string
  ): Promise<ProjectDraftCopyReservationResult>;
  resumeSchemaRefused(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    generation: number
  ): Promise<ProjectDraftSettlementResult>;
  retargetAcknowledgedCopy(options: RetargetAcknowledgedCopyOptions): Promise<ProjectDraftSettlementResult>;
  settleAcknowledgement(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    sentGeneration: number,
    acknowledgedRevision: number,
    acknowledgedMinimumCanvasSchemaVersion?: number
  ): Promise<ProjectDraftSettlementResult>;
  settleConflict(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    sentGeneration: number,
    conflict: ProjectDraftConflict
  ): Promise<ProjectDraftSettlementResult>;
  settleSchemaRefusal(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    sentGeneration: number,
    refusal: ProjectSchemaRefusal
  ): Promise<ProjectDraftSettlementResult>;
  stage(input: ProjectDraftInput): Promise<ProjectDraftStageResult>;
  startFreshWriter(
    projectId: string,
    editorSessionId: string,
    expectedWriterToken: string | null,
    nextWriterToken: string
  ): Promise<ProjectDraftStartWriterResult>;
}

export const PROJECT_DRAFT_MAX_BYTES = 32 * 1024 * 1024;
export const PROJECT_DRAFT_PAGE_LIMIT = 100;
export const PROJECT_DRAFT_PROJECT_LIMIT = 32;

export const getCopySourceProjectName = (copyProjectName: string): string =>
  copyProjectName.endsWith(' (copy)') ? copyProjectName.slice(0, -' (copy)'.length) : copyProjectName;

const states = new Set(['conflict', 'dirty', 'schema-refused']);
const isPositiveInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 1;
const isNonNegativeInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
const isNonEmptyString = (value: unknown): value is string => typeof value === 'string' && value.length > 0;

type ProjectDraftCandidate = Partial<ProjectDraftBase> & {
  conflict?: { kind?: unknown; serverRevision?: unknown };
  refusal?: Partial<ProjectSchemaRefusal>;
  state?: unknown;
};

const isProjectDraftCandidate = (draft: ProjectDraftCandidate, requireDocument: boolean): boolean => {
  const reservationMetadata = [
    draft.copyDocumentByteSize,
    draft.copyProjectId,
    draft.copyProjectGeneration,
    draft.copyProjectMinimumCanvasSchemaVersion,
    draft.copyProjectName,
  ];
  const hasReservation = reservationMetadata.some((value) => value !== undefined);
  if (
    !(draft.baseRevision === null || isPositiveInteger(draft.baseRevision)) ||
    (draft.baseMinimumCanvasSchemaVersion !== undefined && !isPositiveInteger(draft.baseMinimumCanvasSchemaVersion)) ||
    !isNonNegativeInteger(draft.documentByteSize) ||
    (requireDocument && typeof draft.documentJson !== 'string') ||
    !isPositiveInteger(draft.documentSchemaVersion) ||
    !isNonEmptyString(draft.editorSessionId) ||
    !isNonNegativeInteger(draft.generation) ||
    !isNonEmptyString(draft.projectId) ||
    typeof draft.state !== 'string' ||
    !states.has(draft.state) ||
    typeof draft.updatedAt !== 'number' ||
    !Number.isFinite(draft.updatedAt) ||
    draft.updatedAt < 0 ||
    !isNonEmptyString(draft.writerToken) ||
    (draft.copyProjectId !== undefined && !isNonEmptyString(draft.copyProjectId)) ||
    (draft.copyDocumentByteSize !== undefined && !isNonNegativeInteger(draft.copyDocumentByteSize)) ||
    (requireDocument && hasReservation !== isNonEmptyString(draft.copyDocumentJson)) ||
    (draft.copyProjectGeneration !== undefined && !isNonNegativeInteger(draft.copyProjectGeneration)) ||
    (draft.copyProjectMinimumCanvasSchemaVersion !== undefined &&
      !isPositiveInteger(draft.copyProjectMinimumCanvasSchemaVersion)) ||
    (draft.copyProjectName !== undefined && !isNonEmptyString(draft.copyProjectName)) ||
    (draft.copySourceProjectName !== undefined && !isNonEmptyString(draft.copySourceProjectName)) ||
    (hasReservation && reservationMetadata.some((value) => value === undefined))
  ) {
    return false;
  }
  if (draft.state === 'conflict') {
    return (
      draft.refusal === undefined &&
      (draft.conflict?.kind === 'deleted' ||
        (draft.conflict?.kind === 'revision' && isPositiveInteger(draft.conflict.serverRevision)))
    );
  }
  if (draft.state === 'schema-refused') {
    return (
      draft.conflict === undefined &&
      ((draft.refusal?.kind === 'canvas' &&
        isPositiveInteger(draft.refusal.maxCanvasSchemaVersion) &&
        isPositiveInteger(draft.refusal.minimumCanvasSchemaVersion)) ||
        (draft.refusal?.kind === 'document' &&
          isPositiveInteger(draft.refusal.maxDocumentSchemaVersion) &&
          isPositiveInteger(draft.refusal.documentSchemaVersion)) ||
        draft.refusal?.kind === 'invalid-server-document')
    );
  }
  return draft.conflict === undefined && draft.refusal === undefined;
};

export const isProjectDraft = (value: unknown): value is ProjectDraft =>
  Boolean(value && typeof value === 'object' && isProjectDraftCandidate(value as ProjectDraftCandidate, true));

export const isProjectDraftInput = (value: unknown): value is ProjectDraftInput => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const input = value as Partial<ProjectDraftInput>;
  return (
    (input.baseRevision === null || isPositiveInteger(input.baseRevision)) &&
    (input.baseMinimumCanvasSchemaVersion === undefined || isPositiveInteger(input.baseMinimumCanvasSchemaVersion)) &&
    typeof input.documentJson === 'string' &&
    isPositiveInteger(input.documentSchemaVersion) &&
    isNonEmptyString(input.editorSessionId) &&
    isNonNegativeInteger(input.generation) &&
    isNonEmptyString(input.projectId) &&
    typeof input.updatedAt === 'number' &&
    Number.isFinite(input.updatedAt) &&
    input.updatedAt >= 0 &&
    isNonEmptyString(input.writerToken)
  );
};

export const isProjectDraftMetadata = (value: unknown): value is ProjectDraftMetadata =>
  Boolean(
    value &&
    typeof value === 'object' &&
    isProjectDraftCandidate(value as ProjectDraftCandidate, false) &&
    isPositiveInteger((value as Partial<ProjectDraftMetadata>).metadataRevision)
  );

export const isProjectDraftBody = (value: unknown): value is ProjectDraftBody => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const body = value as Partial<ProjectDraftBody>;
  return (
    body.recordType === 'draft-body' &&
    (body.copyDocumentJson === undefined || isNonEmptyString(body.copyDocumentJson)) &&
    isNonNegativeInteger(body.documentByteSize) &&
    isNonEmptyString(body.projectId) &&
    isNonEmptyString(body.editorSessionId) &&
    isNonNegativeInteger(body.generation) &&
    typeof body.documentJson === 'string'
  );
};

export const isProjectDraftWriterClaim = (value: unknown): value is ProjectDraftWriterClaim => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const claim = value as Partial<{
    adoptedByEditorSessionId: string;
    editorSessionId: string;
    fenceReason: 'corrupt-cleanup' | 'moved';
    metadataRevision: number;
    projectId: string;
    retargetedToProjectId: string;
    retargetedToRevision: number;
    state: ProjectDraftWriterClaim['state'];
    updatedAt: number;
    writerToken: string;
  }>;
  const hasValidRetarget =
    (claim.retargetedToProjectId === undefined && claim.retargetedToRevision === undefined) ||
    (isNonEmptyString(claim.retargetedToProjectId) && isPositiveInteger(claim.retargetedToRevision));
  const hasValidState =
    (claim.state === 'active' &&
      claim.adoptedByEditorSessionId === undefined &&
      claim.fenceReason === undefined &&
      claim.retargetedToProjectId === undefined &&
      claim.retargetedToRevision === undefined) ||
    (claim.state === 'fenced' &&
      ((claim.fenceReason === 'moved' && isNonEmptyString(claim.adoptedByEditorSessionId) && hasValidRetarget) ||
        (claim.fenceReason === 'corrupt-cleanup' &&
          claim.adoptedByEditorSessionId === undefined &&
          claim.retargetedToProjectId === undefined &&
          claim.retargetedToRevision === undefined)));
  return (
    hasValidState &&
    isNonEmptyString(claim.projectId) &&
    isNonEmptyString(claim.editorSessionId) &&
    isNonEmptyString(claim.writerToken) &&
    isPositiveInteger(claim.metadataRevision) &&
    typeof claim.updatedAt === 'number' &&
    Number.isFinite(claim.updatedAt) &&
    claim.updatedAt >= 0
  );
};

export const getUtf8ByteSize = (value: string): number => {
  let bytes = 0;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    if (code < 0x80) {
      bytes += 1;
    } else if (code < 0x800) {
      bytes += 2;
    } else if (code >= 0xd800 && code <= 0xdbff && index + 1 < value.length) {
      const next = value.charCodeAt(index + 1);
      if (next >= 0xdc00 && next <= 0xdfff) {
        bytes += 4;
        index += 1;
      } else {
        bytes += 3;
      }
    } else {
      bytes += 3;
    }
  }
  return bytes;
};

export const clampProjectDraftLimit = (value: number | undefined, maximum: number): number =>
  Number.isSafeInteger(value) && value !== undefined && value > 0 ? Math.min(value, maximum) : maximum;

export const toProjectDraftMetadata = (draft: ProjectDraft, metadataRevision = 1): ProjectDraftMetadata => {
  const { copyDocumentJson: _copyDocumentJson, documentJson: _documentJson, ...metadata } = draft;
  return { ...metadata, metadataRevision };
};

export const toProjectDraftBody = (draft: ProjectDraft): ProjectDraftBody => ({
  ...(draft.copyDocumentJson === undefined ? {} : { copyDocumentJson: draft.copyDocumentJson }),
  documentByteSize: draft.documentByteSize,
  documentJson: draft.documentJson,
  editorSessionId: draft.editorSessionId,
  generation: draft.generation,
  projectId: draft.projectId,
  recordType: 'draft-body',
});

export const doProjectDraftPartsMatch = (metadata: ProjectDraftMetadata, body: ProjectDraftBody): boolean =>
  metadata.projectId === body.projectId &&
  metadata.editorSessionId === body.editorSessionId &&
  metadata.generation === body.generation &&
  metadata.documentByteSize === body.documentByteSize &&
  (metadata.copyDocumentByteSize === undefined
    ? body.copyDocumentJson === undefined
    : metadata.copyDocumentByteSize === getUtf8ByteSize(body.copyDocumentJson ?? ''));

export const combineProjectDraft = (metadata: unknown, body: unknown): ProjectDraft | null => {
  if (!isProjectDraftMetadata(metadata) || !isProjectDraftBody(body)) {
    return null;
  }
  if (!doProjectDraftPartsMatch(metadata, body)) {
    return null;
  }
  if (metadata.documentByteSize !== getUtf8ByteSize(body.documentJson)) {
    return null;
  }
  const { metadataRevision: _metadataRevision, ...draftMetadata } = metadata;
  const draft = {
    ...draftMetadata,
    ...(body.copyDocumentJson === undefined ? {} : { copyDocumentJson: body.copyDocumentJson }),
    documentJson: body.documentJson,
  } as ProjectDraft;
  return isProjectDraft(draft) ? draft : null;
};

export const getProjectDraftSummary = (record: unknown, key: ProjectDraftKey): ProjectDraftSummary => {
  if (isProjectDraftMetadata(record)) {
    return {
      documentByteSize: record.documentByteSize,
      documentSchemaVersion: record.documentSchemaVersion,
      editorSessionId: record.editorSessionId,
      generation: record.generation,
      projectId: record.projectId,
      state: record.state,
      updatedAt: record.updatedAt,
    };
  }
  return {
    documentByteSize: null,
    documentSchemaVersion: null,
    editorSessionId: key[1],
    generation: null,
    projectId: key[0],
    state: 'corrupt',
    updatedAt: null,
  };
};

export const isSameProjectDraftGeneration = (draft: ProjectDraft, input: ProjectDraftInput): boolean =>
  draft.documentJson === input.documentJson &&
  draft.documentSchemaVersion === input.documentSchemaVersion &&
  draft.editorSessionId === input.editorSessionId &&
  draft.generation === input.generation &&
  draft.projectId === input.projectId &&
  draft.writerToken === input.writerToken;

export const toDirtyProjectDraft = (draft: ProjectDraft, changes: Partial<ProjectDraftBase>): ProjectDraft => {
  const next: Record<string, unknown> = { ...draft, ...changes, state: 'dirty' };
  delete next.conflict;
  delete next.refusal;
  return next as unknown as ProjectDraft;
};

export const toConflictProjectDraft = (draft: ProjectDraft, conflict: ProjectDraftConflict): ProjectDraft => {
  const next: Record<string, unknown> = { ...draft, conflict, state: 'conflict' };
  delete next.refusal;
  return next as unknown as ProjectDraft;
};

export const toSchemaRefusedProjectDraft = (draft: ProjectDraft, refusal: ProjectSchemaRefusal): ProjectDraft => {
  const next: Record<string, unknown> = { ...draft, refusal, state: 'schema-refused' };
  delete next.conflict;
  return next as unknown as ProjectDraft;
};

export const createUnavailableProjectDraftStore = (): ProjectDraftStore => ({
  availability: 'unavailable',
  acknowledgeRetarget: () => Promise.resolve({ kind: 'unavailable' }),
  adopt: () => Promise.resolve({ kind: 'unavailable' }),
  claimWriter: () => Promise.resolve({ kind: 'unavailable' }),
  close: () => undefined,
  delete: () => Promise.resolve({ kind: 'unavailable' }),
  deleteCorrupt: () => Promise.resolve({ kind: 'unavailable' }),
  get: () => Promise.resolve({ kind: 'unavailable' }),
  list: () => Promise.resolve({ kind: 'unavailable' }),
  listForProject: () => Promise.resolve({ kind: 'unavailable' }),
  listRetargets: () => Promise.resolve({ kind: 'unavailable' }),
  reserveCopyIdentity: () => Promise.resolve({ kind: 'unavailable' }),
  resumeSchemaRefused: () => Promise.resolve({ kind: 'unavailable' }),
  retargetAcknowledgedCopy: () => Promise.resolve({ kind: 'unavailable' }),
  settleAcknowledgement: () => Promise.resolve({ kind: 'unavailable' }),
  settleConflict: () => Promise.resolve({ kind: 'unavailable' }),
  settleSchemaRefusal: () => Promise.resolve({ kind: 'unavailable' }),
  stage: () => Promise.resolve({ kind: 'unavailable' }),
  startFreshWriter: () => Promise.resolve({ kind: 'unavailable' }),
});

const cloneDraft = (draft: ProjectDraft): ProjectDraft => structuredClone(draft);
const draftKey = (projectId: string, editorSessionId: string): string => `${projectId}\u0000${editorSessionId}`;
const compareKeys = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

export const createMemoryProjectDraftStore = ({
  maxDraftBytes = PROJECT_DRAFT_MAX_BYTES,
}: { maxDraftBytes?: number } = {}): ProjectDraftStore => {
  const records = new Map<string, ProjectDraft>();
  const writerClaims = new Map<string, ProjectDraftWriterClaim>();
  let isClosed = false;

  const readRecord = (projectId: string, editorSessionId: string): ProjectDraft | undefined =>
    records.get(draftKey(projectId, editorSessionId));
  const readWriterClaim = (projectId: string, editorSessionId: string): ProjectDraftWriterClaim | undefined =>
    writerClaims.get(draftKey(projectId, editorSessionId));
  const ownsLineage = (projectId: string, editorSessionId: string, writerToken: string): boolean => {
    const claim = readWriterClaim(projectId, editorSessionId);
    return claim?.state === 'active' && claim.writerToken === writerToken;
  };
  const bumpWriterClaim = (key: string): void => {
    const claim = writerClaims.get(key);
    if (claim) {
      writerClaims.set(key, { ...claim, metadataRevision: claim.metadataRevision + 1, updatedAt: Date.now() });
    }
  };
  const writeDraft = (draft: ProjectDraft): ProjectDraft => {
    records.set(draftKey(draft.projectId, draft.editorSessionId), cloneDraft(draft));
    return cloneDraft(draft);
  };
  const settle = (
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    sentGeneration: number,
    transform: (draft: ProjectDraft) => ProjectDraft,
    kind: 'marked' | 'rebased'
  ): ProjectDraftSettlementResult => {
    if (isClosed) {
      return { kind: 'unavailable' };
    }
    const record = readRecord(projectId, editorSessionId);
    if (record === undefined) {
      return { kind: 'missing' };
    }
    if (!ownsLineage(projectId, editorSessionId, writerToken) || record.writerToken !== writerToken) {
      return { kind: 'fenced' };
    }
    if (record.generation < sentGeneration) {
      return { kind: 'stale' };
    }
    const draft = writeDraft(transform(record));
    bumpWriterClaim(draftKey(projectId, editorSessionId));
    return { draft, kind };
  };

  return {
    get availability() {
      return isClosed ? 'unavailable' : 'available';
    },
    acknowledgeRetarget(projectId, editorSessionId, targetProjectId) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const key = draftKey(projectId, editorSessionId);
      const claim = writerClaims.get(key);
      if (
        claim?.state !== 'fenced' ||
        claim.retargetedToProjectId !== targetProjectId ||
        claim.retargetedToRevision === undefined
      ) {
        return Promise.resolve({ kind: 'stale' });
      }
      writerClaims.delete(key);
      return Promise.resolve({ kind: 'deleted' });
    },
    adopt(projectId, fromEditorSessionId, toEditorSessionId, toWriterToken) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const sourceKey = draftKey(projectId, fromEditorSessionId);
      const targetKey = draftKey(projectId, toEditorSessionId);
      const source = records.get(sourceKey);
      if (source === undefined) {
        return Promise.resolve({ kind: 'missing' });
      }
      if (!ownsLineage(projectId, fromEditorSessionId, source.writerToken)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      const targetClaim = writerClaims.get(targetKey);
      if (records.has(targetKey) || (targetClaim?.state === 'active' && targetClaim.writerToken !== toWriterToken)) {
        return Promise.resolve({ kind: 'occupied' });
      }
      records.set(targetKey, { ...cloneDraft(source), editorSessionId: toEditorSessionId, writerToken: toWriterToken });
      records.delete(sourceKey);
      const sourceClaim = writerClaims.get(sourceKey)!;
      const metadataRevision = Math.max(targetClaim?.metadataRevision ?? 0, sourceClaim.metadataRevision) + 1;
      writerClaims.set(targetKey, {
        editorSessionId: toEditorSessionId,
        metadataRevision,
        projectId,
        state: 'active',
        updatedAt: Date.now(),
        writerToken: toWriterToken,
      });
      writerClaims.set(sourceKey, {
        adoptedByEditorSessionId: toEditorSessionId,
        editorSessionId: fromEditorSessionId,
        fenceReason: 'moved',
        metadataRevision,
        projectId,
        state: 'fenced',
        updatedAt: Date.now(),
        writerToken: source.writerToken,
      });
      return Promise.resolve({ kind: 'adopted' });
    },
    claimWriter(projectId, editorSessionId, expectedWriterToken, nextWriterToken) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const key = draftKey(projectId, editorSessionId);
      const claim = writerClaims.get(key);
      const record = records.get(key);
      if (!claim) {
        return Promise.resolve({ kind: 'missing' });
      }
      if (claim.state === 'fenced' || claim.writerToken !== expectedWriterToken) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (record && record.writerToken !== expectedWriterToken) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      if (record) {
        writeDraft({ ...record, writerToken: nextWriterToken });
      }
      writerClaims.set(key, {
        ...claim,
        metadataRevision: claim.metadataRevision + 1,
        updatedAt: Date.now(),
        writerToken: nextWriterToken,
      });
      return Promise.resolve({ kind: 'claimed' });
    },
    close() {
      isClosed = true;
    },
    delete(projectId, editorSessionId, writerToken) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const key = draftKey(projectId, editorSessionId);
      const record = records.get(key);
      const claim = writerClaims.get(key);
      if (claim && (claim.state === 'fenced' || claim.writerToken !== writerToken)) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (record && (!claim || record.writerToken !== writerToken)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      records.delete(key);
      bumpWriterClaim(key);
      return Promise.resolve({ kind: 'deleted' });
    },
    deleteCorrupt() {
      return Promise.resolve({ kind: isClosed ? 'unavailable' : 'not-corrupt' });
    },
    get(projectId, editorSessionId) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const record = readRecord(projectId, editorSessionId);
      if (record === undefined) {
        const claim = readWriterClaim(projectId, editorSessionId);
        if (
          claim?.state === 'fenced' &&
          claim.retargetedToProjectId !== undefined &&
          claim.retargetedToRevision !== undefined
        ) {
          return Promise.resolve({
            kind: 'retargeted',
            projectId: claim.retargetedToProjectId,
            revision: claim.retargetedToRevision,
            writerToken: claim.writerToken,
          });
        }
        if (claim) {
          return Promise.resolve({ kind: 'empty', writerState: claim.state, writerToken: claim.writerToken });
        }
        return Promise.resolve({ kind: 'missing' });
      }
      if (!ownsLineage(projectId, editorSessionId, record.writerToken)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      return Promise.resolve({ draft: cloneDraft(record), kind: 'found' });
    },
    list({ after, limit: requestedLimit } = {}) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PAGE_LIMIT);
      const keys = [...records.values()]
        .map((record): ProjectDraftKey => [record.projectId, record.editorSessionId])
        .sort(
          ([aProject, aSession], [bProject, bSession]) =>
            compareKeys(aProject, bProject) || compareKeys(aSession, bSession)
        );
      const foundStart = after
        ? keys.findIndex(
            ([projectId, sessionId]) => projectId > after[0] || (projectId === after[0] && sessionId > after[1])
          )
        : 0;
      const start = foundStart === -1 ? keys.length : foundStart;
      const pageKeys = keys.slice(start, start + limit);
      const hasMore = start + pageKeys.length < keys.length;
      return Promise.resolve({
        items: pageKeys.map((key) =>
          getProjectDraftSummary(toProjectDraftMetadata(records.get(draftKey(...key)) as ProjectDraft), key)
        ),
        kind: 'available',
        nextCursor: hasMore ? (pageKeys.at(-1) ?? null) : null,
      });
    },
    listForProject(projectId, { after, limit: requestedLimit } = {}) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PROJECT_LIMIT);
      const candidates = [...records.values()]
        .filter((record) => record.projectId === projectId)
        .map((record) =>
          getProjectDraftSummary(toProjectDraftMetadata(record), [record.projectId, record.editorSessionId])
        )
        .sort((a, b) => compareKeys(a.editorSessionId, b.editorSessionId));
      const foundStart = after ? candidates.findIndex((candidate) => candidate.editorSessionId > after) : 0;
      const start = foundStart === -1 ? candidates.length : foundStart;
      const items = candidates.slice(start, start + limit);
      const hasMore = start + items.length < candidates.length;
      return Promise.resolve({
        items,
        kind: 'available',
        nextCursor: hasMore ? (items.at(-1)?.editorSessionId ?? null) : null,
      });
    },
    listRetargets({ after, limit: requestedLimit } = {}) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const limit = clampProjectDraftLimit(requestedLimit, PROJECT_DRAFT_PAGE_LIMIT);
      const matching = [...writerClaims.values()]
        .flatMap((claim): ProjectDraftRetargetHandoff[] =>
          claim.state === 'fenced' &&
          claim.retargetedToProjectId !== undefined &&
          claim.retargetedToRevision !== undefined
            ? [
                {
                  editorSessionId: claim.editorSessionId,
                  projectId: claim.projectId,
                  revision: claim.retargetedToRevision,
                  targetProjectId: claim.retargetedToProjectId,
                  updatedAt: claim.updatedAt,
                },
              ]
            : []
        )
        .sort(
          (a, b) =>
            a.projectId.localeCompare(b.projectId) ||
            a.editorSessionId.localeCompare(b.editorSessionId) ||
            a.targetProjectId.localeCompare(b.targetProjectId)
        );
      const start = after
        ? matching.findIndex(
            (item) =>
              item.projectId > after[0] ||
              (item.projectId === after[0] &&
                (item.editorSessionId > after[1] ||
                  (item.editorSessionId === after[1] && item.targetProjectId > after[2])))
          )
        : 0;
      const normalizedStart = start === -1 ? matching.length : start;
      const items = matching.slice(normalizedStart, normalizedStart + limit);
      const hasMore = normalizedStart + items.length < matching.length;
      const last = items.at(-1);
      return Promise.resolve({
        items,
        kind: 'available',
        nextCursor: hasMore && last ? [last.projectId, last.editorSessionId, last.targetProjectId] : null,
      });
    },
    reserveCopyIdentity(projectId, editorSessionId, writerToken, proposed, replaceCopyProjectId) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const record = readRecord(projectId, editorSessionId);
      if (record === undefined) {
        return Promise.resolve({ kind: 'missing' });
      }
      if (!ownsLineage(projectId, editorSessionId, writerToken) || record.writerToken !== writerToken) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (replaceCopyProjectId !== undefined && record.copyProjectId !== replaceCopyProjectId) {
        return Promise.resolve({ kind: 'stale' });
      }
      const reservation =
        replaceCopyProjectId === undefined && record.copyProjectId
          ? {
              copyDocumentByteSize: record.copyDocumentByteSize!,
              copyDocumentJson: record.copyDocumentJson!,
              copyProjectGeneration: record.copyProjectGeneration!,
              copyProjectId: record.copyProjectId,
              copyProjectMinimumCanvasSchemaVersion: record.copyProjectMinimumCanvasSchemaVersion!,
              copyProjectName: record.copyProjectName!,
              copySourceProjectName: record.copySourceProjectName ?? getCopySourceProjectName(record.copyProjectName!),
            }
          : proposed;
      writeDraft({ ...record, ...reservation });
      bumpWriterClaim(draftKey(projectId, editorSessionId));
      return Promise.resolve({ ...reservation, kind: 'reserved' });
    },
    resumeSchemaRefused(projectId, editorSessionId, writerToken, generation) {
      return Promise.resolve(
        settle(projectId, editorSessionId, writerToken, generation, (draft) => toDirtyProjectDraft(draft, {}), 'marked')
      );
    },
    retargetAcknowledgedCopy(options) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const sourceKey = draftKey(options.projectId, options.editorSessionId);
      const targetKey = draftKey(options.copyProjectId, options.editorSessionId);
      const record = records.get(sourceKey);
      if (record === undefined) {
        const sourceClaim = writerClaims.get(sourceKey);
        if (
          sourceClaim?.state === 'fenced' &&
          sourceClaim.retargetedToProjectId === options.copyProjectId &&
          sourceClaim.retargetedToRevision === options.acknowledgedRevision
        ) {
          if (sourceClaim.writerToken !== options.writerToken) {
            return Promise.resolve({ kind: 'fenced' });
          }
          const target = records.get(targetKey);
          const targetClaim = writerClaims.get(targetKey);
          if (targetClaim && (targetClaim.state !== 'active' || targetClaim.writerToken !== options.writerToken)) {
            return Promise.resolve({ kind: 'fenced' });
          }
          if (target) {
            return Promise.resolve(
              target.writerToken === options.writerToken && targetClaim
                ? { draft: cloneDraft(target), kind: 'retargeted' }
                : { kind: 'corrupt' }
            );
          }
          return Promise.resolve(targetClaim ? { draft: null, kind: 'retargeted' } : { kind: 'corrupt' });
        }
        return Promise.resolve({ kind: 'missing' });
      }
      if (
        !ownsLineage(options.projectId, options.editorSessionId, options.writerToken) ||
        record.writerToken !== options.writerToken
      ) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (record.generation < options.sentGeneration || record.copyProjectId !== options.copyProjectId) {
        return Promise.resolve({ kind: 'stale' });
      }
      const targetClaim = writerClaims.get(targetKey);
      const sourceClaim = writerClaims.get(sourceKey)!;
      if (
        records.has(targetKey) ||
        (targetClaim?.state === 'active' && targetClaim.writerToken !== options.writerToken)
      ) {
        return Promise.resolve({ kind: 'occupied' });
      }
      let retargeted: { documentByteSize: number; documentJson: string } | null = null;
      if (record.generation > options.sentGeneration) {
        let documentJson: string;
        try {
          documentJson = options.retargetDocument(record.documentJson);
        } catch {
          return Promise.resolve({ kind: 'corrupt' });
        }
        const documentByteSize = getUtf8ByteSize(documentJson);
        if (documentByteSize > maxDraftBytes) {
          return Promise.resolve({ kind: 'too-large' });
        }
        retargeted = { documentByteSize, documentJson };
      }
      const draft =
        retargeted === null
          ? null
          : toDirtyProjectDraft(record, {
              baseRevision: options.acknowledgedRevision,
              copyDocumentByteSize: undefined,
              copyDocumentJson: undefined,
              copyProjectId: undefined,
              copyProjectGeneration: undefined,
              copyProjectMinimumCanvasSchemaVersion: undefined,
              copyProjectName: undefined,
              copySourceProjectName: undefined,
              ...retargeted,
              projectId: options.copyProjectId,
            });
      if (draft) {
        records.set(targetKey, cloneDraft(draft));
      }
      records.delete(sourceKey);
      const metadataRevision = Math.max(targetClaim?.metadataRevision ?? 0, sourceClaim.metadataRevision) + 1;
      writerClaims.set(targetKey, {
        editorSessionId: options.editorSessionId,
        metadataRevision,
        projectId: options.copyProjectId,
        state: 'active',
        updatedAt: Date.now(),
        writerToken: options.writerToken,
      });
      writerClaims.set(sourceKey, {
        adoptedByEditorSessionId: options.editorSessionId,
        editorSessionId: options.editorSessionId,
        fenceReason: 'moved',
        metadataRevision,
        projectId: options.projectId,
        retargetedToProjectId: options.copyProjectId,
        retargetedToRevision: options.acknowledgedRevision,
        state: 'fenced',
        updatedAt: Date.now(),
        writerToken: record.writerToken,
      });
      return Promise.resolve({ draft: draft ? cloneDraft(draft) : null, kind: 'retargeted' });
    },
    settleAcknowledgement(
      projectId,
      editorSessionId,
      writerToken,
      sentGeneration,
      acknowledgedRevision,
      acknowledgedMinimumCanvasSchemaVersion
    ) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const record = readRecord(projectId, editorSessionId);
      if (record === undefined) {
        return Promise.resolve({ kind: 'missing' });
      }
      if (!ownsLineage(projectId, editorSessionId, writerToken) || record.writerToken !== writerToken) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (record.generation <= sentGeneration) {
        records.delete(draftKey(projectId, editorSessionId));
        bumpWriterClaim(draftKey(projectId, editorSessionId));
        return Promise.resolve({ kind: 'deleted' });
      }
      const draft = writeDraft({
        ...record,
        baseMinimumCanvasSchemaVersion: acknowledgedMinimumCanvasSchemaVersion ?? record.baseMinimumCanvasSchemaVersion,
        baseRevision: acknowledgedRevision,
      });
      bumpWriterClaim(draftKey(projectId, editorSessionId));
      return Promise.resolve({ draft, kind: 'rebased' });
    },
    settleConflict(projectId, editorSessionId, writerToken, sentGeneration, conflict) {
      return Promise.resolve(
        settle(
          projectId,
          editorSessionId,
          writerToken,
          sentGeneration,
          (draft) => toConflictProjectDraft(draft, conflict),
          'marked'
        )
      );
    },
    settleSchemaRefusal(projectId, editorSessionId, writerToken, sentGeneration, refusal) {
      return Promise.resolve(
        settle(
          projectId,
          editorSessionId,
          writerToken,
          sentGeneration,
          (draft) => toSchemaRefusedProjectDraft(draft, refusal),
          'marked'
        )
      );
    },
    stage(input) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      if (!isProjectDraftInput(input)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      const documentByteSize = getUtf8ByteSize(input.documentJson);
      if (documentByteSize > maxDraftBytes) {
        return Promise.resolve({ kind: 'too-large' });
      }
      const current = readRecord(input.projectId, input.editorSessionId);
      const key = draftKey(input.projectId, input.editorSessionId);
      const claim = writerClaims.get(key);
      if (claim && (claim.state === 'fenced' || claim.writerToken !== input.writerToken)) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (current && (!claim || current.writerToken !== input.writerToken)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      if (current) {
        if (current.generation > input.generation) {
          return Promise.resolve({ kind: 'stale' });
        }
        if (current.generation === input.generation) {
          return Promise.resolve({
            kind: isSameProjectDraftGeneration(current, input) ? 'replayed' : 'generation-conflict',
          });
        }
        writeDraft({
          ...current,
          documentByteSize,
          documentJson: input.documentJson,
          documentSchemaVersion: input.documentSchemaVersion,
          generation: input.generation,
          updatedAt: input.updatedAt,
        });
        bumpWriterClaim(key);
        return Promise.resolve({ kind: 'stored' });
      }
      const draft: ProjectDraft = { ...input, documentByteSize, state: 'dirty' };
      if (!isProjectDraft(draft)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      writerClaims.set(key, {
        editorSessionId: input.editorSessionId,
        metadataRevision: (claim?.metadataRevision ?? 0) + 1,
        projectId: input.projectId,
        state: 'active',
        updatedAt: input.updatedAt,
        writerToken: input.writerToken,
      });
      writeDraft(draft);
      return Promise.resolve({ kind: 'stored' });
    },
    startFreshWriter(projectId, editorSessionId, expectedWriterToken, nextWriterToken) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const key = draftKey(projectId, editorSessionId);
      if (records.has(key)) {
        return Promise.resolve({ kind: 'occupied' });
      }
      const claim = writerClaims.get(key);
      if (claim ? claim.writerToken !== expectedWriterToken : expectedWriterToken !== null) {
        return Promise.resolve({ kind: 'fenced' });
      }
      writerClaims.set(key, {
        editorSessionId,
        metadataRevision: (claim?.metadataRevision ?? 0) + 1,
        projectId,
        state: 'active',
        updatedAt: Date.now(),
        writerToken: nextWriterToken,
      });
      return Promise.resolve({ kind: 'started' });
    },
  };
};
