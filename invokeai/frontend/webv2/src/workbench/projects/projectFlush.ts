/**
 * What a push to the server actually achieved, as distinct from whether the call threw.
 *
 * Its own module rather than part of the sync engine: the caller that needs it, `library.ts`, is a
 * module the sync engine itself imports, so reaching back would be a cycle.
 *
 * A push swallows every failure because they are all recoverable — the document is cached and the
 * next save retries. But "recoverable" is not "done", and reading a project back after an
 * unacknowledged push is how an export ships someone's work minus the last ten minutes of it.
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

/** Refuse anything the server has not certainly acknowledged. See {@link readAcknowledgedProject}. */
export const assertProjectFlushed = (outcome: ProjectPushOutcome): void => {
  if (outcome.kind !== 'acknowledged') {
    throw new ProjectFlushError(outcome.kind);
  }
};

/**
 * Who a recovery fork *is*, separately from what it holds. `recoveredProject` is built from the
 * document as it was when the push started, so applying it wholesale overwrites anything edited
 * since; a project still open adopts these four fields instead and keeps its live content. The
 * snapshot stays for the case with no live project left to re-identify — a tab closed mid-flight.
 */
export interface ProjectRecoveredIdentity {
  id: string;
  name: string;
  recoveredAt: string;
  recoveryOf: string;
}
