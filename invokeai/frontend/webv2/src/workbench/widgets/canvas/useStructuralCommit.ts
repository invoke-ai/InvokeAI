import type {
  CanvasDocumentCapability,
  CanvasLayerBasePatch,
  CanvasLayerCapability,
  CanvasLayerConfigPatch,
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
  readonly layers: Pick<CanvasLayerCapability, 'commitPrepared' | 'endStructuralPreview'>;
}

export type PreparedCommitOutcome =
  | StructuralCommitResult
  | { status: 'refused'; refusal: DocumentRefusal }
  | { status: 'unchanged' };

/**
 * Prepares an edit against the engine's current model and commits it; refusals and no-ops never dispatch. A preview
 * gesture open elsewhere ends first, so the edit and its inverse capture the committed document, not the preview.
 */
export const commitPreparedEdit = (
  engine: CanvasPreparedEngine | null,
  label: string,
  prepare: (model: CanvasDocumentModel) => PrepareEditResult
): PreparedCommitOutcome => {
  engine?.layers.endStructuralPreview();
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
  readonly layers: Pick<CanvasLayerCapability, 'beginStructuralPreview' | 'commitPrepared' | 'endStructuralPreview'>;
}

/**
 * Prepares the gesture's edit. `baseline` restores what the gesture previewed, for the edit's `before`; it is null
 * when nothing was previewed or the engine ended the gesture (a replay or an edit from elsewhere landed), and the
 * edit then records from the live document.
 */
export type PreviewCommit = (
  label: string,
  prepare: (model: CanvasDocumentModel, baseline: CanvasLayerPreviewMutation | null) => PrepareEditResult
) => PreparedCommitOutcome;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * `source` narrowed to the fields `shape` names, record by record, or undefined when `shape` names a field `source`
 * does not hold: the model refuses a `before` whose fields differ from the edit's.
 */
const narrowTo = (
  source: Record<string, unknown>,
  shape: Record<string, unknown>
): Record<string, unknown> | undefined => {
  const narrowed: Record<string, unknown> = {};
  for (const key of Object.keys(shape)) {
    if (!(key in source)) {
      return undefined;
    }
    const have = source[key];
    const want = shape[key];
    if (isRecord(have) && isRecord(want)) {
      const nested = narrowTo(have, want);
      if (!nested) {
        return undefined;
      }
      narrowed[key] = nested;
    } else {
      narrowed[key] = have;
    }
  }
  return narrowed;
};

/** The baseline's values for the fields `patch` names, as a `patch` command's `before`; undefined without them. */
export const baselinePatch = (
  baseline: CanvasLayerPreviewMutation | null,
  patch: CanvasLayerBasePatch
): CanvasLayerBasePatch | undefined =>
  baseline?.type === 'updateCanvasLayer' ? (narrowTo(baseline.patch, patch) as CanvasLayerBasePatch) : undefined;

/** The baseline's values for the fields `config` names, as a `patch-config` command's `before`; undefined without them. */
export const baselineConfig = (
  baseline: CanvasLayerPreviewMutation | null,
  config: CanvasLayerConfigPatch
): CanvasLayerConfigPatch | undefined =>
  baseline?.type === 'updateCanvasLayerConfig'
    ? (narrowTo(baseline.config, config) as CanvasLayerConfigPatch | undefined)
    : undefined;

export interface StructuralPreview {
  /** Previews `action` live, at most once per frame; false while edits are refused. */
  preview(action: CanvasLayerPreviewMutation): boolean;
  /** Records the gesture as one undo step from its baseline and reports any refusal. */
  commit: PreviewCommit;
  /** Ends the gesture unrecorded, restoring what it previewed. */
  cancel(): void;
  /**
   * What the gesture's open preview session replaced, while the document carries its previews; null before the first
   * preview and once the engine ended the session.
   */
  baseline(): CanvasLayerPreviewMutation | null;
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
      const session = sessionRef.current;
      if (session?.apply(action)) {
        return true;
      }
      // No session yet, or the engine ended it: the next session starts from the document as it is now. A session
      // that only refused this preview (edits locked mid-gesture) stays the gesture's, holding its previews and
      // baseline for the commit or cancel to come.
      sessionRef.current = engine.layers.beginStructuralPreview() ?? (session?.isActive() ? session : null);
      return sessionRef.current?.apply(action) ?? false;
    },
    [engine]
  );

  const commit = useCallback<PreviewCommit>(
    (label, prepare) => {
      const session = sessionRef.current;
      sessionRef.current = null;
      // The engine ended the gesture (a replay or an edit from elsewhere landed): its edit commits like any prepared
      // edit, and a target that vanished with that document change is not the user's commit to explain.
      const sessionEnded = session !== null && !session.isActive();
      const model = engine?.document.model() ?? null;
      let outcome: PreparedCommitOutcome;
      if (!engine || !model) {
        session?.cancel();
        outcome = { status: 'not-ready' };
      } else {
        const result = prepare(model, session?.baseline() ?? null);
        if (result.status === 'prepared') {
          const committed = session?.commit(label, result.edit);
          outcome = committed && !sessionEnded ? committed : engine.layers.commitPrepared(label, result.edit);
        } else {
          session?.cancel();
          outcome = result.status === 'unchanged' ? result : { refusal: result, status: 'refused' };
        }
      }
      if (!(sessionEnded && outcome.status === 'refused' && outcome.refusal.status === 'missing')) {
        reportPreparedCommit(outcome, notify.error, t);
      }
      return outcome;
    },
    [engine, notify, t]
  );

  const cancel = useCallback((): void => {
    const session = sessionRef.current;
    sessionRef.current = null;
    session?.cancel();
  }, []);

  const baseline = useCallback((): CanvasLayerPreviewMutation | null => sessionRef.current?.baseline() ?? null, []);

  return { baseline, cancel, commit, preview };
};
