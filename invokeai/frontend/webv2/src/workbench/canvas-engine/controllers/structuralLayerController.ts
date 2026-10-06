import type {
  CanvasLayerPreviewMutation,
  PreparedCommitOptions,
  StructuralCommitOptions,
  StructuralCommitResult,
  StructuralPreviewSession,
} from '@workbench/canvas-engine/capabilities';
import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { PreparedDocumentEdit } from '@workbench/canvas-engine/document-model/documentCommands';
import type { HistoryEntryToken } from '@workbench/canvas-engine/history/history';
import type { CanvasMutationOrigin, CanvasProjectMutation } from '@workbench/canvas-engine/mutationContracts';

import { createDocumentModel, previewInverse } from '@workbench/canvas-engine/document-model/documentModel';
import { checkEditPostconditions } from '@workbench/canvas-engine/document-model/postconditions';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { createDocumentPatchEntry } from '@workbench/canvas-engine/history/documentPatch';
import { HISTORY_ENTRY_OVERHEAD_BYTES } from '@workbench/canvas-engine/history/history';

import { EditStepError, type CanvasMutationContext, type EditStep } from './mutationContext';

/** The edit revision a mutation's insertion was anchored at; a commit refuses an anchor captured earlier. */
const anchorRevisionOf = (mutation: CanvasProjectMutation): number | undefined => {
  switch (mutation.type) {
    case 'addCanvasLayer':
      return mutation.anchor.capturedEditRevision;
    case 'applyCanvasLayerStackMutation':
      return mutation.add?.[0]?.anchor.capturedEditRevision;
    default:
      return undefined;
  }
};

export type StructuralMutationContext = Pick<
  CanvasMutationContext,
  | 'applyStep'
  | 'begin'
  | 'canEdit'
  | 'dispatch'
  | 'getDocument'
  | 'getEditRevision'
  | 'getReducerDocument'
  | 'historyTop'
  | 'isGestureActive'
  | 'projectId'
>;

export type StructuralFailureReport =
  | 'Structural history replay was refused'
  | 'Structural history replay could not be mirrored';

export interface StructuralLayerControllerOptions {
  readonly ctx: StructuralMutationContext;
  readonly getSelectedLayerIds?: (document: CanvasDocumentContractV3) => readonly string[];
  readonly report?: (message: StructuralFailureReport, label: string, error: unknown) => void;
  readonly now?: () => number;
  /** Defers a preview flush to the next paint, returning a cancel. Defaults to rAF, synchronous without one. */
  readonly schedulePreview?: (flush: () => void) => () => void;
}

const defaultSchedulePreview = (flush: () => void): (() => void) => {
  if (typeof requestAnimationFrame !== 'function') {
    flush();
    return () => undefined;
  }
  const frame = requestAnimationFrame(flush);
  return () => cancelAnimationFrame(frame);
};

interface NudgeBurst {
  readonly token: HistoryEntryToken;
  readonly expiresAt: number;
  readonly selectionKey: string;
  readonly origins: readonly { id: string; x: number; y: number }[];
}

const NUDGE_COALESCE_MS = 500;

const positionMutation = (positions: readonly { id: string; x: number; y: number }[]): CanvasProjectMutation => ({
  type: 'setCanvasLayerPositions',
  updates: [...positions],
});

const unique = (ids: readonly string[]): string[] => [...new Set(ids)];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** `existing` keeps every value it holds; `incoming` adds the rest, field by field inside plain records. */
const mergeRecords = (
  existing: Record<string, unknown>,
  incoming: Record<string, unknown>
): Record<string, unknown> => {
  const merged: Record<string, unknown> = { ...incoming, ...existing };
  for (const key of Object.keys(incoming)) {
    const have = existing[key];
    const add = incoming[key];
    if (isRecord(have) && isRecord(add)) {
      merged[key] = mergeRecords(have, add);
    }
  }
  return merged;
};

/** The session baseline with `inverse` added for the fields it does not hold yet; earlier captures win. */
const withBaseline = (
  baseline: CanvasLayerPreviewMutation | null,
  inverse: CanvasLayerPreviewMutation
): CanvasLayerPreviewMutation => {
  if (!baseline) {
    return inverse;
  }
  if (baseline.type === 'updateCanvasLayer' && inverse.type === 'updateCanvasLayer') {
    return { ...baseline, patch: mergeRecords(baseline.patch, inverse.patch) as typeof baseline.patch };
  }
  if (baseline.type === 'updateCanvasLayerConfig' && inverse.type === 'updateCanvasLayerConfig') {
    return { ...baseline, config: mergeRecords(baseline.config, inverse.config) as typeof baseline.config };
  }
  return baseline;
};

type RecordedResult = StructuralCommitResult | { status: 'committed'; token: HistoryEntryToken };

/** The public result; the history token stays internal. */
const publicResult = (result: RecordedResult): StructuralCommitResult =>
  result.status === 'committed' ? { status: 'committed' } : result;

/** Owns guarded, failure-atomic structural document edits, nudge coalescing and live preview sessions. */
export class StructuralLayerController {
  private burst: NudgeBurst | null = null;
  private disposed = false;
  private readonly now: () => number;
  private preview: { session: StructuralPreviewSession; drop(): void; end(): void } | null = null;

  constructor(private readonly deps: StructuralLayerControllerOptions) {
    this.now = deps.now ?? Date.now;
  }

  canCommit(): boolean {
    const { ctx } = this.deps;
    return !this.disposed && ctx.canEdit() && !ctx.isGestureActive();
  }

  commit(
    label: string,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    options: StructuralCommitOptions = {}
  ): StructuralCommitResult {
    const stale = this.staleRevision(
      options.expectedRevision ?? anchorRevisionOf(forward) ?? anchorRevisionOf(inverse)
    );
    if (stale) {
      return stale;
    }
    this.endPreview();
    return publicResult(this.publish(label, this.step(forward, inverse, options.verify), forward, inverse));
  }

  /** A prepared flat edit: refused as `stale` unless its revision and project still match, verified by its postconditions. */
  commitPrepared(
    label: string,
    edit: PreparedDocumentEdit,
    options: PreparedCommitOptions = {}
  ): StructuralCommitResult {
    const stale = this.staleRevision(edit.expectedRevision);
    if (stale) {
      return stale;
    }
    if (edit.projectId !== this.deps.ctx.projectId) {
      return { status: 'dispatch-rejected' };
    }
    this.endPreview();
    const step = this.step(edit.forward, edit.inverse, (document) =>
      checkEditPostconditions(document, edit.postconditions)
    );
    if (edit.history !== 'record') {
      return this.applyUnrecorded(step);
    }
    return publicResult(this.publish(label, step, edit.forward, edit.inverse, options.origin));
  }

  /**
   * Starts an owned preview: `apply` coalesces live dispatches to one per frame and captures the values the first
   * preview of each field replaces, `commit` records the gesture as one step from its prepared baseline without
   * dispatching again when the preview already reached it, restoring the baseline when it is refused. A newer
   * session, any commit, a history replay or disposal ends this one first, restoring its baseline.
   */
  beginPreview(): StructuralPreviewSession | null {
    if (!this.canCommit()) {
      return null;
    }
    this.endPreview();
    let pending: CanvasProjectMutation | null = null;
    let cancelFlush: (() => void) | null = null;
    let baseline: CanvasLayerPreviewMutation | null = null;
    const owns = (): boolean => this.preview?.session === session;
    const flushPending = (): void => {
      cancelFlush?.();
      cancelFlush = null;
      const next = pending;
      pending = null;
      if (next && this.canCommit()) {
        this.deps.ctx.dispatch(next, 'system');
      }
    };
    // Returning to the baseline undoes this session's own unrecorded previews, so a lock does not block it.
    const restore = (): void => {
      if (baseline && !this.disposed) {
        this.deps.ctx.dispatch(baseline, 'system');
      }
    };
    const drop = (): void => {
      cancelFlush?.();
      cancelFlush = null;
      pending = null;
    };
    const end = (): void => {
      drop();
      restore();
    };
    const session: StructuralPreviewSession = {
      apply: (action: CanvasLayerPreviewMutation) => {
        if (!owns() || !this.canCommit()) {
          return false;
        }
        // A session previews one mutation kind on one node; its baseline holds each field as the gesture found it.
        if (baseline && (baseline.id !== action.id || baseline.type !== action.type)) {
          return false;
        }
        const document = this.deps.ctx.getReducerDocument();
        const node = document ? getDocumentIndex(document).byId.get(action.id)?.node : undefined;
        if (!node) {
          return false;
        }
        baseline = withBaseline(baseline, previewInverse(node, action));
        pending = action;
        if (cancelFlush === null) {
          let flushed = false;
          const cancel = (this.deps.schedulePreview ?? defaultSchedulePreview)(() => {
            flushed = true;
            cancelFlush = null;
            flushPending();
          });
          cancelFlush = flushed ? null : cancel;
        }
        return true;
      },
      baseline: () => (owns() ? baseline : null),
      cancel: () => {
        if (owns()) {
          this.endPreview();
        }
      },
      commit: (label, edit) => {
        if (!owns()) {
          return { status: 'busy' };
        }
        flushPending();
        this.preview = null;
        if (edit.projectId !== this.deps.ctx.projectId) {
          restore();
          return { status: 'dispatch-rejected' };
        }
        const verify = (document: CanvasDocumentContractV3 | null): boolean =>
          document !== null && checkEditPostconditions(document, edit.postconditions);
        // The preview already published the final values: record the step without another reducer allocation.
        const step: EditStep = verify(this.deps.ctx.getReducerDocument())
          ? {}
          : this.step(edit.forward, edit.inverse, verify);
        const result = publicResult(this.publish(label, step, edit.forward, edit.inverse));
        // A refused gesture must not leave its unrecorded previews behind.
        if (result.status !== 'committed') {
          restore();
        }
        return result;
      },
      isActive: owns,
    };
    this.preview = { drop, end, session };
    return session;
  }

  /** Ends an open preview session: pending previews are dropped and its baseline is restored, unrecorded. */
  endPreview(): void {
    const preview = this.preview;
    if (!preview) {
      return;
    }
    this.preview = null;
    preview.end();
  }

  /** Forgets an open preview session without restoring it: the document it previewed on is gone. */
  dropPreview(): void {
    const preview = this.preview;
    if (!preview) {
      return;
    }
    this.preview = null;
    preview.drop();
  }

  nudge(dx: number, dy: number): StructuralCommitResult {
    const { ctx } = this.deps;
    // The nudge is prepared from the committed document, like any other edit.
    this.endPreview();
    const document = ctx.getDocument();
    if (this.disposed || !document?.selectedLayerId) {
      return { status: this.disposed ? 'not-ready' : 'dispatch-rejected' };
    }
    const ids = unique([document.selectedLayerId, ...(this.deps.getSelectedLayerIds?.(document) ?? [])]);
    const model = createDocumentModel(document, { editRevision: ctx.getEditRevision(), projectId: ctx.projectId });
    const prepared = model.prepare({ dx, dy, ids, type: 'translate' });
    if (prepared.status !== 'prepared' || prepared.edit.forward.type !== 'setCanvasLayerPositions') {
      return { status: 'dispatch-rejected' };
    }
    const leaves = model.compileLeaves();
    if (prepared.edit.touchedIds.some((id) => !leaves.find((leaf) => leaf.id === id)?.contributionEnabled)) {
      return { status: 'dispatch-rejected' };
    }
    const selectionKey = prepared.edit.touchedIds.join('\0');
    const now = this.now();
    // Coalesce only into the step this burst recorded: any interleaved entry starts a fresh one.
    const burst = this.burst;
    const coalesce =
      burst !== null &&
      burst.token === ctx.historyTop() &&
      burst.selectionKey === selectionKey &&
      now < burst.expiresAt;
    const origins = coalesce
      ? burst.origins
      : (prepared.edit.inverse as Extract<CanvasProjectMutation, { type: 'setCanvasLayerPositions' }>).updates;
    const forward = prepared.edit.forward;
    const inverse = positionMutation(origins);
    const result = this.publish(
      'Nudge layer',
      this.step(forward, inverse, (next) => checkEditPostconditions(next, prepared.edit.postconditions)),
      forward,
      inverse,
      undefined,
      coalesce ? burst.token : undefined
    );
    this.burst =
      'token' in result ? { expiresAt: now + NUDGE_COALESCE_MS, origins, selectionKey, token: result.token } : null;
    return publicResult(result);
  }

  dispose(): void {
    this.disposed = true;
    this.burst = null;
    this.dropPreview();
  }

  private staleRevision(expectedRevision: number | undefined): StructuralCommitResult | null {
    const actualRevision = this.deps.ctx.getEditRevision();
    return expectedRevision !== undefined && expectedRevision !== actualRevision
      ? { actualRevision, expectedRevision, status: 'stale' }
      : null;
  }

  /** A step that dispatches `forward`, verifies it, and dispatches `inverse` back when verification fails. */
  private step(
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    verify?: (document: CanvasDocumentContractV3) => boolean
  ): EditStep {
    const { ctx } = this.deps;
    const before = ctx.getReducerDocument();
    return {
      accepted: (document) => document !== null && document !== before && (verify?.(document) ?? true),
      mutation: forward,
      rollback: { mutation: inverse, restored: () => true },
    };
  }

  private publish(
    label: string,
    step: EditStep,
    forward: CanvasProjectMutation,
    inverse: CanvasProjectMutation,
    origin?: CanvasMutationOrigin,
    replacing?: HistoryEntryToken
  ): RecordedResult {
    if (this.disposed) {
      return { status: 'not-ready' };
    }
    const txn = this.deps.ctx.begin({ historyBytes: HISTORY_ENTRY_OVERHEAD_BYTES });
    if (!('publish' in txn)) {
      return { status: txn.status };
    }
    try {
      const entry = createDocumentPatchEntry({
        dispatch: (action) => this.replay(label, action, action === forward ? inverse : forward),
        forward,
        inverse,
        label,
      });
      const published = txn.publish(label, step, entry, { origin, replacing, routed: forward });
      return published.status === 'committed' ? { status: 'committed', token: published.token } : published;
    } finally {
      txn.end();
    }
  }

  private applyUnrecorded(step: EditStep): StructuralCommitResult {
    const { ctx } = this.deps;
    if (this.disposed || !ctx.canEdit()) {
      return { status: this.disposed ? 'not-ready' : 'busy' };
    }
    if (ctx.isGestureActive()) {
      return { status: 'gesture-active' };
    }
    try {
      ctx.applyStep(step);
      return { status: 'committed' };
    } catch (error) {
      if (!(error instanceof EditStepError)) {
        throw error;
      }
      return error.outcome === 'rejected'
        ? { status: 'dispatch-rejected' }
        : { recovered: error.outcome, status: 'postcondition-failed' };
    }
  }

  /**
   * A reducer that refuses a replay (its target changed since) is expected: the entry moves as a no-op and the
   * refusal is reported. A mirror that cannot follow an accepted replay is not: the opposite action restores the
   * reducer and the step stays where it was.
   */
  private replay(label: string, action: CanvasProjectMutation, opposite: CanvasProjectMutation): void {
    try {
      this.deps.ctx.applyStep({ mutation: action, rollback: { mutation: opposite, restored: () => true } });
    } catch (error) {
      if (error instanceof EditStepError && error.outcome === 'rejected') {
        this.deps.report?.('Structural history replay was refused', label, error);
        return;
      }
      this.deps.report?.('Structural history replay could not be mirrored', label, error);
      throw error;
    }
  }
}
