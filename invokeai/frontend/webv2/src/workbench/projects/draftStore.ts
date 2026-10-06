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

/**
 * The lineage's writer has settled its unload journal through `generation`: the server or the lineage's draft holds a
 * document at least that new, or the writer discarded it. Entries of that writer at or below it are obsolete even
 * when they outlive their deletion. A mark of an earlier writer means nothing. Optional, so claims written before it
 * existed read as having settled nothing.
 */
export interface ProjectUnloadJournalSettlement {
  generation: number;
  writerToken: string;
}

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
      unloadJournalSettled?: ProjectUnloadJournalSettlement;
      updatedAt: number;
      writerToken: string;
    }
  | {
      editorSessionId: string;
      metadataRevision: number;
      projectId: string;
      state: 'active';
      unloadJournalSettled?: ProjectUnloadJournalSettlement;
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

/**
 * A document a writer had not staged when its page was hidden or unloaded. Staging takes fenced read-then-write
 * transactions that cannot finish while a page unloads; this record is written blind instead, keyed by its writer and
 * generation so it overwrites nothing else, and becomes a draft only through reconciliation.
 */
export interface ProjectUnloadJournalEntry {
  /** The account whose lifetime wrote it; the database is already per account, so this is defence in depth. */
  accountId: string;
  baseMinimumCanvasSchemaVersion?: number;
  baseRevision: number | null;
  documentByteSize: number;
  documentJson: string;
  documentSchemaVersion: number;
  editorSessionId: string;
  /**
   * The draft generation the writer reserved for this document: above every generation it staged before and below
   * every one it stages after (or that of a stage of the same document still in flight), so the lineage alone tells
   * whether the journal is still the newest copy.
   */
  generation: number;
  journaledAt: number;
  projectId: string;
  recordType: 'unload-journal';
  schemaVersion: typeof PROJECT_UNLOAD_JOURNAL_SCHEMA_VERSION;
  writerToken: string;
}
export type ProjectUnloadJournalKey = [
  projectId: string,
  editorSessionId: string,
  writerToken: string,
  generation: number,
];
export interface ProjectUnloadJournalWrite {
  entries: readonly ProjectUnloadJournalEntry[];
  /** Each writer lineage's entries through the key's generation are deleted: what they held is in recovery now. */
  retired: readonly ProjectUnloadJournalKey[];
}
export type ProjectUnloadJournalWriteResult =
  /** `written` settles true once the write commits, false if it aborted. */
  { kind: 'started'; written: Promise<boolean> } | { kind: 'unavailable' };
export type ProjectUnloadJournalOutcome =
  | 'applied'
  | 'corrupt'
  | 'expired'
  | 'fenced'
  | 'foreign-account'
  | 'invalid'
  | 'live'
  | 'quota'
  | 'settled'
  | 'superseded';
export interface ProjectUnloadJournalReconciliation {
  editorSessionId: string;
  outcome: ProjectUnloadJournalOutcome;
  projectId: string;
  /** For an applied entry, the draft it replaced: if the server holds that document, it holds this writer's save. */
  replacedDraft?: { documentJson: string; generation: number };
}
export type ProjectUnloadJournalReconcileResult =
  | { kind: 'available'; outcomes: ProjectUnloadJournalReconciliation[] }
  | { kind: 'unavailable' };
export interface ProjectUnloadJournalReconcileOptions {
  /**
   * True for a lineage whose editor session a page other than this one still holds. Its entries are that page's to
   * settle (its save acknowledges or retires them) and are left in place, outcome `'live'`, until it is gone.
   */
  isEditorSessionLive?: (editorSessionId: string) => Promise<boolean>;
}
export type ProjectUnloadJournalSettleResult = { kind: 'corrupt' | 'fenced' | 'quota' | 'settled' | 'unavailable' };
export type ProjectUnloadJournalDecision =
  | { draft: ProjectDraft; kind: 'apply' }
  | { kind: 'discard'; reason: 'expired' | 'fenced' | 'foreign-account' | 'invalid' | 'settled' | 'superseded' }
  /** The lineage is damaged; the entry waits for its cleanup, for at most the retention period. */
  | { kind: 'retain'; reason: 'corrupt' };

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
  /** With `settledThroughGeneration`, also settles the writer's unload journal through it, in the same transaction. */
  delete(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    settledThroughGeneration?: number
  ): Promise<ProjectDraftDeleteResult>;
  deleteCorrupt(projectId: string, editorSessionId: string): Promise<ProjectDraftCorruptDeleteResult>;
  /**
   * Removes one writer's unload journal entries for a lineage, through `throughGeneration` (all when omitted). A blind
   * range delete: it reads nothing, so it never holds up a journal write of a page that is unloading.
   */
  discardUnloadJournal(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    throughGeneration?: number
  ): Promise<{ kind: 'deleted' | 'unavailable' }>;
  get(projectId: string, editorSessionId: string): Promise<ProjectDraftGetResult>;
  /**
   * Synchronously starts one blind, explicitly committed write: the only kind that outlives a page unload. Each entry
   * replaces its writer's lower generations for the lineage; `retired` lineages lose theirs. It reads nothing, so it
   * cannot check fencing; reconciliation does.
   */
  journalBeforeUnload(write: ProjectUnloadJournalWrite): ProjectUnloadJournalWriteResult;
  list(options?: { after?: ProjectDraftKey; limit?: number }): Promise<ProjectDraftPageResult>;
  listForProject(projectId: string, options?: { after?: string; limit?: number }): Promise<ProjectDraftListResult>;
  listRetargets(options?: {
    after?: ProjectDraftRetargetCursor;
    limit?: number;
  }): Promise<ProjectDraftRetargetListResult>;
  /** The projects with unload journal entries, for deciding whether anything would open. */
  peekUnloadJournalProjectIds(
    limit: number
  ): Promise<{ kind: 'available'; projectIds: string[] } | { kind: 'unavailable' }>;
  /**
   * Turns each unload journal entry into its lineage's newest draft, or discards it, by `decideUnloadJournalEntry`:
   * one fenced draft transaction per entry, then the entry's removal. An entry left behind by a crash in between is
   * superseded by the draft it produced, so running again is safe. Run before any writer of this load claims a lineage.
   * Entries of a lineage whose editor session is still live elsewhere are skipped: a page that is merely hidden is
   * still writing that lineage, and staging its journal would hand another tab an edit it may yet undo.
   */
  reconcileUnloadJournal(
    accountId: string,
    now: number,
    options?: ProjectUnloadJournalReconcileOptions
  ): Promise<ProjectUnloadJournalReconcileResult>;
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
  /** Also settles the writer's unload journal through `sentGeneration`, in the same transaction. */
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
  /**
   * Settles the writer's unload journal of a lineage through `throughGeneration` (claiming an unclaimed lineage for it),
   * then deletes those entries. `settled` once the mark commits; the deletion is best effort, the mark makes it moot.
   */
  settleUnloadJournal(
    projectId: string,
    editorSessionId: string,
    writerToken: string,
    throughGeneration: number
  ): Promise<ProjectUnloadJournalSettleResult>;
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
export const PROJECT_UNLOAD_JOURNAL_SCHEMA_VERSION = 1;
/**
 * The most document bytes one hidden-page journal writes. Serializing and cloning the documents is synchronous in the
 * `pagehide` handler; above this a document is left to the ordinary autosave.
 */
export const PROJECT_UNLOAD_JOURNAL_MAX_BYTES = 8 * 1024 * 1024;
/** How long an entry for a damaged lineage waits for that lineage's cleanup before it is discarded. */
export const PROJECT_UNLOAD_JOURNAL_RETENTION_MS = 7 * 24 * 60 * 60 * 1000;

/** The claim with its writer's unload journal settled through `generation`; the mark never moves back. */
export const withUnloadJournalSettled = <Claim extends ProjectDraftWriterClaim>(
  claim: Claim,
  generation: number
): Claim => {
  const settled =
    claim.unloadJournalSettled?.writerToken === claim.writerToken ? claim.unloadJournalSettled.generation : 0;
  return {
    ...claim,
    unloadJournalSettled: { generation: Math.max(settled, generation), writerToken: claim.writerToken },
  };
};

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
    unloadJournalSettled: Partial<ProjectUnloadJournalSettlement>;
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
  const hasValidSettlement =
    claim.unloadJournalSettled === undefined ||
    (typeof claim.unloadJournalSettled === 'object' &&
      claim.unloadJournalSettled !== null &&
      isNonNegativeInteger(claim.unloadJournalSettled.generation) &&
      isNonEmptyString(claim.unloadJournalSettled.writerToken));
  return (
    hasValidState &&
    hasValidSettlement &&
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

const isProjectUnloadJournalEntry = (
  value: unknown,
  maxDocumentBytes = PROJECT_DRAFT_MAX_BYTES
): value is ProjectUnloadJournalEntry => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const entry = value as Partial<ProjectUnloadJournalEntry>;
  return (
    entry.recordType === 'unload-journal' &&
    entry.schemaVersion === PROJECT_UNLOAD_JOURNAL_SCHEMA_VERSION &&
    isNonEmptyString(entry.accountId) &&
    (entry.baseRevision === null || isPositiveInteger(entry.baseRevision)) &&
    (entry.baseMinimumCanvasSchemaVersion === undefined || isPositiveInteger(entry.baseMinimumCanvasSchemaVersion)) &&
    isPositiveInteger(entry.documentSchemaVersion) &&
    isNonEmptyString(entry.editorSessionId) &&
    isPositiveInteger(entry.generation) &&
    typeof entry.journaledAt === 'number' &&
    Number.isFinite(entry.journaledAt) &&
    entry.journaledAt >= 0 &&
    isNonEmptyString(entry.projectId) &&
    isNonEmptyString(entry.writerToken) &&
    typeof entry.documentJson === 'string' &&
    isNonNegativeInteger(entry.documentByteSize) &&
    entry.documentByteSize <= maxDocumentBytes &&
    entry.documentByteSize === getUtf8ByteSize(entry.documentJson)
  );
};

/**
 * The reconciliation rules for one unload journal entry, given its lineage as the draft store holds it. The entry is
 * staged exactly as its writer would have staged it, so it is only applied where that writer could still write: the
 * lineage is unclaimed or claimed by the same writer, which has not settled its journal that far, and holds nothing at
 * or above the entry's generation. Sticky conflict, schema refusal, copy reservation and base revision of an existing
 * draft are kept, as staging keeps them.
 */
export const decideUnloadJournalEntry = ({
  accountId,
  claim,
  current,
  entry,
  maxDocumentBytes = PROJECT_DRAFT_MAX_BYTES,
  now,
}: {
  accountId: string;
  /** `'corrupt'` for a stored claim that does not validate. */
  claim: ProjectDraftWriterClaim | 'corrupt' | undefined;
  /** The lineage's draft, `null` when it holds none, `'corrupt'` when its records do not agree with the claim. */
  current: ProjectDraft | 'corrupt' | null;
  entry: unknown;
  maxDocumentBytes?: number;
  now: number;
}): ProjectUnloadJournalDecision => {
  if (!isProjectUnloadJournalEntry(entry, maxDocumentBytes)) {
    return { kind: 'discard', reason: 'invalid' };
  }
  if (entry.accountId !== accountId) {
    return { kind: 'discard', reason: 'foreign-account' };
  }
  const retainForCleanup = (): ProjectUnloadJournalDecision =>
    now - entry.journaledAt > PROJECT_UNLOAD_JOURNAL_RETENTION_MS
      ? { kind: 'discard', reason: 'expired' }
      : { kind: 'retain', reason: 'corrupt' };
  if (claim === 'corrupt') {
    return retainForCleanup();
  }
  // Another editor adopted, retargeted, or reclaimed the lineage after this writer lost it: never write over that.
  if (claim && (claim.state === 'fenced' || claim.writerToken !== entry.writerToken)) {
    return { kind: 'discard', reason: 'fenced' };
  }
  // Acknowledged, already in recovery, or discarded; after an acknowledgement the lineage may hold no draft at all.
  if (
    claim?.unloadJournalSettled?.writerToken === entry.writerToken &&
    claim.unloadJournalSettled.generation >= entry.generation
  ) {
    return { kind: 'discard', reason: 'settled' };
  }
  if (current === 'corrupt') {
    return retainForCleanup();
  }
  if (current && current.generation >= entry.generation) {
    return { kind: 'discard', reason: 'superseded' };
  }
  const documentFields = {
    documentByteSize: entry.documentByteSize,
    documentJson: entry.documentJson,
    documentSchemaVersion: entry.documentSchemaVersion,
    generation: entry.generation,
    updatedAt: entry.journaledAt,
  };
  const draft: ProjectDraft = current
    ? ({ ...current, ...documentFields } as ProjectDraft)
    : {
        ...(entry.baseMinimumCanvasSchemaVersion === undefined
          ? {}
          : { baseMinimumCanvasSchemaVersion: entry.baseMinimumCanvasSchemaVersion }),
        ...documentFields,
        baseRevision: entry.baseRevision,
        editorSessionId: entry.editorSessionId,
        projectId: entry.projectId,
        state: 'dirty',
        writerToken: entry.writerToken,
      };
  return isProjectDraft(draft) ? { draft, kind: 'apply' } : { kind: 'discard', reason: 'invalid' };
};

export const createUnavailableProjectDraftStore = (): ProjectDraftStore => ({
  availability: 'unavailable',
  acknowledgeRetarget: () => Promise.resolve({ kind: 'unavailable' }),
  adopt: () => Promise.resolve({ kind: 'unavailable' }),
  claimWriter: () => Promise.resolve({ kind: 'unavailable' }),
  close: () => undefined,
  delete: () => Promise.resolve({ kind: 'unavailable' }),
  deleteCorrupt: () => Promise.resolve({ kind: 'unavailable' }),
  discardUnloadJournal: () => Promise.resolve({ kind: 'unavailable' }),
  get: () => Promise.resolve({ kind: 'unavailable' }),
  journalBeforeUnload: () => ({ kind: 'unavailable' }),
  peekUnloadJournalProjectIds: () => Promise.resolve({ kind: 'unavailable' }),
  list: () => Promise.resolve({ kind: 'unavailable' }),
  listForProject: () => Promise.resolve({ kind: 'unavailable' }),
  listRetargets: () => Promise.resolve({ kind: 'unavailable' }),
  reconcileUnloadJournal: () => Promise.resolve({ kind: 'unavailable' }),
  settleUnloadJournal: () => Promise.resolve({ kind: 'unavailable' }),
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
const journalWriterKey = (projectId: string, editorSessionId: string, writerToken: string): string =>
  `${draftKey(projectId, editorSessionId)}\u0000${writerToken}`;
const compareKeys = (left: string, right: string): number => (left < right ? -1 : left > right ? 1 : 0);

export const createMemoryProjectDraftStore = ({
  maxDraftBytes = PROJECT_DRAFT_MAX_BYTES,
}: { maxDraftBytes?: number } = {}): ProjectDraftStore => {
  const records = new Map<string, ProjectDraft>();
  const writerClaims = new Map<string, ProjectDraftWriterClaim>();
  const unloadJournal = new Map<string, ProjectUnloadJournalEntry>();
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
  const deleteJournalThrough = (writerKey: string, generation: number, inclusive: boolean): void => {
    for (const [entryKey, entry] of unloadJournal) {
      if (
        journalWriterKey(entry.projectId, entry.editorSessionId, entry.writerToken) === writerKey &&
        (inclusive ? entry.generation <= generation : entry.generation < generation)
      ) {
        unloadJournal.delete(entryKey);
      }
    }
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
    delete(projectId, editorSessionId, writerToken, settledThroughGeneration) {
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
      const bumped = writerClaims.get(key);
      if (bumped && settledThroughGeneration !== undefined) {
        writerClaims.set(key, withUnloadJournalSettled(bumped, settledThroughGeneration));
      }
      return Promise.resolve({ kind: 'deleted' });
    },
    deleteCorrupt() {
      return Promise.resolve({ kind: isClosed ? 'unavailable' : 'not-corrupt' });
    },
    discardUnloadJournal(projectId, editorSessionId, writerToken, throughGeneration = Number.POSITIVE_INFINITY) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      deleteJournalThrough(journalWriterKey(projectId, editorSessionId, writerToken), throughGeneration, true);
      return Promise.resolve({ kind: 'deleted' });
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
    journalBeforeUnload({ entries, retired }) {
      if (isClosed) {
        return { kind: 'unavailable' };
      }
      for (const [projectId, editorSessionId, writerToken, generation] of retired) {
        deleteJournalThrough(journalWriterKey(projectId, editorSessionId, writerToken), generation, true);
      }
      for (const entry of entries) {
        const writerKey = journalWriterKey(entry.projectId, entry.editorSessionId, entry.writerToken);
        deleteJournalThrough(writerKey, entry.generation, false);
        unloadJournal.set(`${writerKey}\u0000${entry.generation}`, structuredClone(entry));
      }
      return { kind: 'started', written: Promise.resolve(true) };
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
    peekUnloadJournalProjectIds(limit) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      return Promise.resolve({
        kind: 'available',
        projectIds: [...new Set([...unloadJournal.values()].map((entry) => entry.projectId))].slice(0, limit),
      });
    },
    async reconcileUnloadJournal(accountId, now, { isEditorSessionLive } = {}) {
      if (isClosed) {
        return { kind: 'unavailable' };
      }
      const outcomes: ProjectUnloadJournalReconciliation[] = [];
      const ordered = [...unloadJournal].sort(
        ([leftKey, left], [rightKey, right]) =>
          compareKeys(
            leftKey.slice(0, leftKey.lastIndexOf('\u0000')),
            rightKey.slice(0, rightKey.lastIndexOf('\u0000'))
          ) || left.generation - right.generation
      );
      const liveEditorSessions = new Map<string, boolean>();
      for (const [entryKey, entry] of ordered) {
        if (isEditorSessionLive) {
          let isLive = liveEditorSessions.get(entry.editorSessionId);
          if (isLive === undefined) {
            isLive = await isEditorSessionLive(entry.editorSessionId);
            liveEditorSessions.set(entry.editorSessionId, isLive);
          }
          if (isLive) {
            outcomes.push({ editorSessionId: entry.editorSessionId, outcome: 'live', projectId: entry.projectId });
            continue;
          }
        }
        const key = draftKey(entry.projectId, entry.editorSessionId);
        const claim = writerClaims.get(key);
        const record = records.get(key);
        const decision = decideUnloadJournalEntry({
          accountId,
          claim,
          current:
            record === undefined
              ? null
              : claim?.state === 'active' && claim.writerToken === record.writerToken
                ? record
                : 'corrupt',
          entry,
          maxDocumentBytes: maxDraftBytes,
          now,
        });
        const outcome: ProjectUnloadJournalReconciliation = {
          editorSessionId: entry.editorSessionId,
          outcome: decision.kind === 'apply' ? 'applied' : decision.reason,
          projectId: entry.projectId,
        };
        if (decision.kind === 'apply') {
          writerClaims.set(
            key,
            claim
              ? { ...claim, metadataRevision: claim.metadataRevision + 1, updatedAt: entry.journaledAt }
              : {
                  editorSessionId: entry.editorSessionId,
                  metadataRevision: 1,
                  projectId: entry.projectId,
                  state: 'active',
                  updatedAt: entry.journaledAt,
                  writerToken: entry.writerToken,
                }
          );
          writeDraft(decision.draft);
          if (record) {
            outcome.replacedDraft = { documentJson: record.documentJson, generation: record.generation };
          }
        }
        if (decision.kind !== 'retain') {
          unloadJournal.delete(entryKey);
        }
        outcomes.push(outcome);
      }
      return { kind: 'available', outcomes };
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
      const ownClaim = readWriterClaim(projectId, editorSessionId);
      if (ownClaim?.state === 'active' && ownClaim.writerToken === writerToken) {
        writerClaims.set(draftKey(projectId, editorSessionId), withUnloadJournalSettled(ownClaim, sentGeneration));
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
    settleUnloadJournal(projectId, editorSessionId, writerToken, throughGeneration) {
      if (isClosed) {
        return Promise.resolve({ kind: 'unavailable' });
      }
      const key = draftKey(projectId, editorSessionId);
      const claim = writerClaims.get(key);
      if (claim && (claim.state === 'fenced' || claim.writerToken !== writerToken)) {
        return Promise.resolve({ kind: 'fenced' });
      }
      if (!claim && records.has(key)) {
        return Promise.resolve({ kind: 'corrupt' });
      }
      writerClaims.set(
        key,
        withUnloadJournalSettled(
          claim ?? {
            editorSessionId,
            metadataRevision: 1,
            projectId,
            state: 'active',
            updatedAt: Date.now(),
            writerToken,
          },
          throughGeneration
        )
      );
      deleteJournalThrough(journalWriterKey(projectId, editorSessionId, writerToken), throughGeneration, true);
      return Promise.resolve({ kind: 'settled' });
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
