/**
 * Read models of the intermediates manager. Pure: no transport, React or UI imports.
 *
 * An intermediate is classified once by the server under the cleanup policy: `safe` (deleted by either mode),
 * `referenced` (named by a saved document or persisted editor state; deleted only by a force clear), `active` (produced or
 * consumed by queued or running work) and `recent` (inside the grace window). The last two are never deleted.
 */

export type IntermediatesCleanupMode = 'safe' | 'force';

export type IntermediatesSummarySort = 'reclaimable_bytes' | 'project_name';

export type IntermediatesOperationStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface IntermediatesKindCounts {
  safe: number;
  referenced: number;
  active: number;
  recent: number;
}

export interface IntermediatesScopeTarget {
  userId: string;
  /** `null` is the owner's unassigned intermediates: no project, or a project that no longer exists. */
  projectId: string | null;
}

export interface IntermediatesRow extends IntermediatesScopeTarget {
  userDisplayName: string | null;
  userEmail: string | null;
  projectName: string | null;
  /** The newest durable image on the project's board, for a thumbnail; null for unassigned rows. */
  coverImageName: string | null;
  images: IntermediatesKindCounts;
  videos: IntermediatesKindCounts;
  /** Measured bytes of the safe items. */
  reclaimableBytes: number;
  /** Measured bytes a force clear adds. */
  referencedBytes: number;
  /** Safe or referenced items not yet measured. */
  unknownSizeCount: number;
}

export interface IntermediatesSummaryTotals {
  rows: number;
  safeImages: number;
  safeVideos: number;
  inUseImages: number;
  inUseVideos: number;
  reclaimableBytes: number;
  unknownSizeCount: number;
}

export interface IntermediatesSummary {
  items: IntermediatesRow[];
  /** Rows matching the request across every page. */
  total: number;
  offset: number;
  limit: number;
  /** Over every matching row, not only the returned page. */
  totals: IntermediatesSummaryTotals;
  recentGraceSeconds: number;
  /** Sizes are still being measured in the background. */
  measuring: boolean;
  canManageEveryone: boolean;
}

/** Every row the summary filters match minus `excluded`, resolved by the server so no page needs enumerating. */
export interface IntermediatesMatchingScope {
  kind: 'matching';
  /** The owner filter; null is the caller's account, or every account for an administrator. */
  userId: string | null;
  projectId: string | null;
  search: string | null;
  excluded: IntermediatesScopeTarget[];
}

export type IntermediatesScope =
  | { kind: 'selection'; targets: IntermediatesScopeTarget[] }
  | { kind: 'owner'; userId: string }
  | { kind: 'everyone' }
  | IntermediatesMatchingScope;

export interface IntermediatesImpact {
  deleteImages: number;
  deleteVideos: number;
  keepReferencedImages: number;
  keepReferencedVideos: number;
  keepActiveImages: number;
  keepActiveVideos: number;
  keepRecentImages: number;
  keepRecentVideos: number;
  reclaimableBytes: number;
  unknownSizeCount: number;
}

/**
 * What names the media: a saved project or library workflow, the legacy editor's persisted state (`ownerId` is its
 * state key), or a project set aside for repair by an earlier migration.
 */
export type IntermediatesAffectedDocumentKind = 'project' | 'workflow' | 'client_state' | 'quarantined_project';

export interface IntermediatesAffectedDocument {
  kind: IntermediatesAffectedDocumentKind;
  /** The document's owner. */
  userId: string;
  userDisplayName: string | null;
  userEmail: string | null;
  ownerId: string;
  name: string | null;
  references: number;
}

export interface IntermediatesPreview {
  previewId: string;
  mode: IntermediatesCleanupMode;
  scope: IntermediatesScope;
  createdAt: string;
  expiresAt: string;
  targetRows: number;
  impact: IntermediatesImpact;
  /** Confirming acknowledges these; a document saved after the preview keeps its media. */
  affectedDocuments: IntermediatesAffectedDocument[];
  /** Every affected document; the server lists only the first of them. */
  affectedDocumentsTotal: number;
}

export interface IntermediatesOperationProgress {
  processedImages: number;
  processedVideos: number;
  deletedImages: number;
  deletedVideos: number;
  retainedImages: number;
  retainedVideos: number;
  failedImages: number;
  failedVideos: number;
  reclaimedBytes: number;
  /** Deleted items whose size was never measured; their bytes are not in `reclaimedBytes`. */
  unknownSizeCount: number;
  pendingDiskCleanup: number;
}

/** A cleanup run. The server keeps it in memory only: a restart forgets it, and requesting the scope again is the retry. */
export interface IntermediatesOperation {
  operationId: string;
  userId: string;
  mode: IntermediatesCleanupMode;
  /** The scope as requested, so the same request can be made again. */
  scope: IntermediatesScope;
  status: IntermediatesOperationStatus;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  error: string | null;
  /** Deletions the preview expected. */
  targetImages: number;
  targetVideos: number;
  progress: IntermediatesOperationProgress;
}

export const getIntermediatesRowKey = (target: IntermediatesScopeTarget): string =>
  `${target.userId}\u0000${target.projectId ?? ''}`;

export const getKindTotal = (counts: IntermediatesKindCounts): number =>
  counts.safe + counts.referenced + counts.active + counts.recent;

export const getKindInUse = (counts: IntermediatesKindCounts): number =>
  counts.referenced + counts.active + counts.recent;

export const getOperationTotalTargets = (operation: IntermediatesOperation): number =>
  operation.targetImages + operation.targetVideos;

export const getOperationProcessed = (operation: IntermediatesOperation): number =>
  operation.progress.processedImages + operation.progress.processedVideos;

export const isOperationSettled = (operation: IntermediatesOperation): boolean =>
  operation.status === 'completed' || operation.status === 'failed';

/** A stopped or partly failed run left work behind that requesting the same scope again picks up. */
export const canRunOperationAgain = (operation: IntermediatesOperation): boolean =>
  isOperationSettled(operation) &&
  (operation.status === 'failed' || operation.progress.failedImages + operation.progress.failedVideos > 0);
