/**
 * Canvas edit guards share {@link CanvasEditConcurrency}. Exclusive operation leases abort on document/project
 * replacement or owned-layer changes. Permits capture the editing-lock epoch and recheck before publishing; lock
 * owners bypass the lock.
 *
 * Structural/pixel commits and history replay refuse active gestures, and every edit is refused while a history
 * step replays; tools own stroke commits, and floats guard lift rather than bake. Project ids and edit revisions
 * fence insertion/structural commits; exports also guard layer reference, cache version and document generation.
 *
 * {@link CanvasTransactionOutcome} describes runtime refusals. Document command refusals remain
 * runtime-independent; combined results use {@link SubsetOf} to verify their vocabulary.
 */
export interface CanvasEditConcurrency {
  readonly projectId: string;
  canEdit(owner?: symbol): boolean;
  capturePermit(owner?: symbol): DocumentEditPermit | null;
  isPermitCurrent(permit: DocumentEditPermit): boolean;
  isGestureActive(): boolean;
  /** Counts reducer document identities; a captured value is stale once any edit lands. */
  getEditRevision(): number;
}

export interface DocumentEditPermit {
  readonly epoch: number;
  readonly owner?: symbol;
}

export type CanvasTransactionOutcome = 'busy' | 'gesture-active' | 'stale' | 'aborted' | 'not-ready' | 'over-budget';

/** Names members of a vocabulary; a member outside `T` is a compile error rather than a silent drop. */
export type SubsetOf<T, U extends T> = U;
