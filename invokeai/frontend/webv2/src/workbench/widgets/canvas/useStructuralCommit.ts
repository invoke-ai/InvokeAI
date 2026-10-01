import type {
  CanvasDocumentCapability,
  CanvasLayerCapability,
  CanvasLayerPreviewMutation,
  CanvasDocumentModel,
  DocumentRefusal,
  MaskEditResult,
  PrepareEditResult,
  StructuralCommitResult,
  StructuralPreviewSession,
} from '@workbench/canvas-engine/api';
import type { TFunction } from 'i18next';

import { useNotify } from '@workbench/useNotify';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

/** Outcomes a layer pixel operation (merge down, rasterize, layer via copy) reports instead of applying. */
export type LayerOperationRefusal =
  | 'busy'
  | 'empty'
  | 'failed'
  | 'locked'
  | 'missing'
  | 'not-ready'
  | 'over-budget'
  | 'stale'
  | 'unsupported';

const LAYER_OPERATION_KEYS: Record<LayerOperationRefusal, string> = {
  busy: 'widgets.layers.actions.busy',
  empty: 'widgets.layers.actions.empty',
  failed: 'widgets.layers.actions.operationFailed',
  locked: 'widgets.layers.actions.locked',
  missing: 'widgets.layers.actions.missing',
  'not-ready': 'widgets.layers.actions.notReady',
  'over-budget': 'widgets.layers.actions.overBudget',
  stale: 'widgets.layers.actions.stale',
  unsupported: 'widgets.layers.actions.unsupported',
};

/** Explains a refused layer pixel operation; nothing changed in the document, pixels or history. */
export const reportLayerOperation = (
  refusal: LayerOperationRefusal,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void => reportError(t('widgets.layers.actions.actionFailed'), t(LAYER_OPERATION_KEYS[refusal]));

/** Explains a refused mask clear or invert; `busy` and a mask with nothing to change stay silent. */
export const reportMaskEdit = (
  result: MaskEditResult,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void => {
  if (result.status !== 'committed' && result.status !== 'busy' && result.status !== 'nothing') {
    reportLayerOperation(result.status, reportError, t);
  }
};

/**
 * Explains a refused structural edit. `busy` stays silent because an in-flight operation already
 * disables the controls that could reach here; every other refusal drops the edit and says so.
 */
export const reportStructuralCommit = (
  result: StructuralCommitResult,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void => {
  const title = t('widgets.canvas.structural.failed');
  switch (result.status) {
    case 'committed':
    case 'busy':
      return;
    case 'gesture-active':
      reportError(title, t('widgets.canvas.structural.gestureActive'));
      return;
    case 'not-ready':
      reportError(title, t('widgets.canvas.structural.notReady'));
      return;
    case 'over-budget':
      reportError(title, t('widgets.canvas.structural.overBudget'));
      return;
    case 'stale':
      reportError(title, t('widgets.canvas.structural.stale'));
      return;
    case 'dispatch-rejected':
      reportError(title, t('widgets.canvas.structural.rejected'));
      return;
    case 'postcondition-failed':
      switch (result.recovered) {
        case 'reverted':
          reportError(title, t('widgets.canvas.structural.reverted'));
          return;
        case 'reverted-unmirrored':
          reportError(title, t('widgets.canvas.structural.unmirrored'));
          return;
        case 'unreverted':
          reportError(title, t('widgets.canvas.structural.unverified'));
      }
  }
};

/** The engine surface a prepared edit needs: the document model and the transaction. */
export interface CanvasPreparedEngine {
  readonly document: Pick<CanvasDocumentCapability, 'model'>;
  readonly layers: Pick<CanvasLayerCapability, 'commitPrepared'>;
}

export type PreparedCommitOutcome =
  | StructuralCommitResult
  | { status: 'refused'; refusal: DocumentRefusal }
  | { status: 'unchanged' };

/** Prepares an edit against the engine's current model and commits it; refusals and no-ops never dispatch. */
export const commitPreparedEdit = (
  engine: CanvasPreparedEngine | null,
  label: string,
  prepare: (model: CanvasDocumentModel) => PrepareEditResult
): PreparedCommitOutcome => {
  const model = engine?.document.model() ?? null;
  if (!engine || !model) {
    return { status: 'not-ready' };
  }
  const result = prepare(model);
  switch (result.status) {
    case 'prepared':
      return engine.layers.commitPrepared(label, result.edit);
    case 'unchanged':
      return result;
    default:
      return { refusal: result, status: 'refused' };
  }
};

const REFUSAL_KEYS: Record<DocumentRefusal['status'], string> = {
  'invalid-target': 'widgets.canvas.structural.refusedInvalidTarget',
  locked: 'widgets.canvas.structural.refusedLocked',
  missing: 'widgets.canvas.structural.refusedMissing',
  unsupported: 'widgets.canvas.structural.refusedUnsupported',
  'wrong-type': 'widgets.canvas.structural.refusedWrongType',
};

const INVALID_TARGET_KEYS: Partial<Record<Extract<DocumentRefusal, { status: 'invalid-target' }>['reason'], string>> = {
  cycle: 'widgets.canvas.structural.refusedCycle',
  'depth-exceeded': 'widgets.canvas.structural.refusedDepth',
  'node-limit': 'widgets.canvas.structural.refusedNodeLimit',
  'not-siblings': 'widgets.canvas.structural.refusedNotSiblings',
};

const refusalKey = (refusal: DocumentRefusal): string =>
  (refusal.status === 'invalid-target' ? INVALID_TARGET_KEYS[refusal.reason] : undefined) ??
  REFUSAL_KEYS[refusal.status];

export const reportPreparedCommit = (
  outcome: PreparedCommitOutcome,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void => {
  if (outcome.status === 'unchanged') {
    return;
  }
  if (outcome.status === 'refused') {
    reportError(t('widgets.canvas.structural.failed'), t(refusalKey(outcome.refusal)));
    return;
  }
  reportStructuralCommit(outcome, reportError, t);
};

export type PreparedCommit = (
  label: string,
  prepare: (model: CanvasDocumentModel) => PrepareEditResult
) => PreparedCommitOutcome;

/** A widget-side prepared commit that reports every refusal the user needs to hear about. */
export const usePreparedCommit = (engine: CanvasPreparedEngine | null): PreparedCommit => {
  const notify = useNotify();
  const { t } = useTranslation();

  return useCallback(
    (label, prepare) => {
      const outcome = commitPreparedEdit(engine, label, prepare);

      reportPreparedCommit(outcome, notify.error, t);
      return outcome;
    },
    [engine, notify, t]
  );
};

export interface CanvasPreviewEngine extends CanvasPreparedEngine {
  readonly layers: Pick<CanvasLayerCapability, 'beginStructuralPreview' | 'commitPrepared'>;
}

export interface StructuralPreview {
  /** Previews `action` live, at most once per frame; false while edits are refused. */
  preview(action: CanvasLayerPreviewMutation): boolean;
  /** Records the gesture as one undo step from its baseline and reports any refusal. */
  commit: PreparedCommit;
  /** Ends the gesture unrecorded, returning the document to `restore`. */
  cancel(restore?: CanvasLayerPreviewMutation): void;
}

/** A live-preview gesture backed by one engine preview session, committed or cancelled as a whole. */
export const useStructuralPreview = (engine: CanvasPreviewEngine | null): StructuralPreview => {
  const notify = useNotify();
  const { t } = useTranslation();
  const sessionRef = useRef<StructuralPreviewSession | null>(null);

  const preview = useCallback(
    (action: CanvasLayerPreviewMutation): boolean => {
      if (!engine) {
        return false;
      }
      if (sessionRef.current?.apply(action)) {
        return true;
      }
      // No session yet, or a newer one or a commit ended it.
      sessionRef.current = engine.layers.beginStructuralPreview();
      return sessionRef.current?.apply(action) ?? false;
    },
    [engine]
  );

  const commit = useCallback<PreparedCommit>(
    (label, prepare) => {
      const session = sessionRef.current;
      sessionRef.current = null;
      const model = engine?.document.model() ?? null;
      let outcome: PreparedCommitOutcome;
      if (!engine || !model) {
        session?.cancel();
        outcome = { status: 'not-ready' };
      } else {
        const result = prepare(model);
        if (result.status === 'prepared') {
          const committed = session?.commit(label, result.edit);
          // A session another gesture ended commits like any prepared edit.
          outcome =
            committed && committed.status !== 'busy' ? committed : engine.layers.commitPrepared(label, result.edit);
        } else {
          session?.cancel();
          outcome = result.status === 'unchanged' ? result : { refusal: result, status: 'refused' };
        }
      }
      reportPreparedCommit(outcome, notify.error, t);
      return outcome;
    },
    [engine, notify, t]
  );

  const cancel = useCallback((restore?: CanvasLayerPreviewMutation): void => {
    const session = sessionRef.current;
    sessionRef.current = null;
    session?.cancel(restore);
  }, []);

  return { cancel, commit, preview };
};
