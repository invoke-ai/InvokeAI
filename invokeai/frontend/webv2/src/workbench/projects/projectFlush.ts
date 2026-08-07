/**
 * What a push to the server actually achieved, as distinct from whether the call threw.
 *
 * Its own module rather than part of the sync engine, because the two callers that need it —
 * `library.ts` for duplication and `projectFile.ts` for export — are modules the sync engine itself
 * imports. Reaching back into it for this would make that a cycle.
 *
 * Every failure the engine can hit is recoverable: the document is cached locally and the next save
 * retries, which is why a push swallows them all rather than rejecting. But "recoverable" is not the
 * same as "done", and reading a project back from the server after an unacknowledged push is how an
 * export or a copy silently ships someone's work minus the last ten minutes of it.
 */

/** The document the push was carrying, so a caller can compare it with what the engine recorded. */
export interface ProjectPushOutcomeBase {
  documentJson: string;
}

export type ProjectPushOutcome =
  /** The server holds exactly this document. */
  | ({ kind: 'acknowledged' } & ProjectPushOutcomeBase)
  /**
   * This project id no longer holds our document: it was deleted or overwritten elsewhere, and the
   * local edits continue under a different id. Reading the id back would read a stranger's version.
   */
  | ({ kind: 'superseded' } & ProjectPushOutcomeBase)
  /** The push did not land. The server still holds whatever it last acknowledged. */
  | ({ kind: 'unsynced' } & ProjectPushOutcomeBase);

/** Raised where an unacknowledged push must not be treated as a successful one. */
export class ProjectFlushError extends Error {
  readonly reason: 'superseded' | 'unsynced';

  constructor(reason: 'superseded' | 'unsynced') {
    super(
      reason === 'unsynced'
        ? 'The project has changes that have not reached the server.'
        : 'The project was replaced on the server; the local edits continue under another id.'
    );
    this.name = 'ProjectFlushError';
    this.reason = reason;
  }
}

/**
 * Read back only what the server has certainly acknowledged.
 *
 * Both callers — export and duplicate — do the same two things: flush the open project, then GET
 * the record. The GET returns the last *acknowledged* document, so without this the flush's failure
 * is indistinguishable from its success and the copy is built from stale bytes under a clean
 * success toast.
 */
export const assertProjectFlushed = (outcome: ProjectPushOutcome): void => {
  if (outcome.kind !== 'acknowledged') {
    throw new ProjectFlushError(outcome.kind);
  }
};

/**
 * Who a recovery fork *is*, separately from what it holds.
 *
 * The fork's `recoveredProject` is built from the document as it was when the push started, so
 * applying it wholesale overwrites anything edited since. The identity is the only part of it the
 * local store actually needs: a project that is still open can adopt these four fields and keep its
 * live content, which is both newer and already what the person is looking at. The snapshot stays
 * for the case where there is no live project left to re-identify — a tab closed mid-flight.
 */
export interface ProjectRecoveredIdentity {
  id: string;
  name: string;
  recoveredAt: string;
  recoveryOf: string;
}
