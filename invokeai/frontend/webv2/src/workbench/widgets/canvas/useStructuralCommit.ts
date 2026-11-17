import type {
  AdjustmentRefusal,
  AdjustmentRefusalReason,
  CanvasDocumentCapability,
  CanvasLayerEditRefusal,
  CanvasLayerBasePatch,
  CanvasLayerCapability,
  CanvasLayerConfigPatch,
  CanvasLayerPreviewMutation,
  CanvasDocumentModel,
  DocumentRefusal,
  LayerUnreadable,
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

const ADJUSTMENT_REFUSAL_KEYS: Record<AdjustmentRefusalReason, string> = {
  error: 'widgets.layers.adjustments.refusal.error',
  'over-budget': 'widgets.layers.adjustments.refusal.overBudget',
  'too-large': 'widgets.layers.adjustments.refusal.tooLarge',
};

/** Names the layer or group whose Clarity, Texture or Dehaze could not run, and why. */
export const describeAdjustmentRefusal = (refusal: AdjustmentRefusal, t: TFunction): string =>
  t(ADJUSTMENT_REFUSAL_KEYS[refusal.reason], { name: refusal.ownerName });

/** Refusal copy per status; only hidden or locked layers offer a new layer, since one above escapes the cause. */
const LAYER_REFUSAL_KEYS: Record<
  CanvasLayerEditRefusal['status'],
  { message: string; offerMessage: string | null; title: string }
> = {
  'erase-unsupported': {
    message: 'widgets.canvas.editRefusal.eraseUnsupportedMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.eraseUnsupported',
  },
  'gradient-unsupported': {
    message: 'widgets.canvas.editRefusal.gradientUnsupportedMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.gradientUnsupported',
  },
  'layer-alpha-locked': {
    message: 'widgets.canvas.editRefusal.alphaLockedMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.alphaLocked',
  },
  'layer-display-hidden': {
    message: 'widgets.canvas.editRefusal.displayHiddenMessage',
    offerMessage: 'widgets.canvas.editRefusal.displayHiddenOfferMessage',
    title: 'widgets.canvas.editRefusal.displayHidden',
  },
  'layer-hidden': {
    message: 'widgets.canvas.editRefusal.hiddenMessage',
    offerMessage: 'widgets.canvas.editRefusal.hiddenOfferMessage',
    title: 'widgets.canvas.editRefusal.hidden',
  },
  'layer-loading': {
    message: 'widgets.canvas.editRefusal.loadingMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.loading',
  },
  'layer-unpaintable': {
    message: 'widgets.canvas.editRefusal.unpaintableMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.unpaintable',
  },
  'layer-locked': {
    message: 'widgets.canvas.editRefusal.lockedMessage',
    offerMessage: 'widgets.canvas.editRefusal.lockedOfferMessage',
    title: 'widgets.canvas.editRefusal.locked',
  },
  'selection-erase-unsupported': {
    message: 'widgets.canvas.editRefusal.selectionEraseUnsupportedMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.selectionEraseUnsupported',
  },
  'transform-session': {
    message: 'widgets.canvas.editRefusal.transformSessionMessage',
    offerMessage: null,
    title: 'widgets.canvas.editRefusal.transformSession',
  },
};

/** An overlay out of generation is turned back on with its dot, not its eye; it has no new-layer offer. */
const OVERLAY_EXCLUDED_KEYS = {
  message: 'widgets.canvas.editRefusal.excludedMessage',
  offerMessage: null,
  title: 'widgets.canvas.editRefusal.excluded',
} as const;

/** The notice for a layer the active tool could not edit, with an offer only for statuses with offer wording. */
export const describeLayerEditRefusal = (
  refusal: CanvasLayerEditRefusal,
  t: TFunction,
  overlay: boolean
): { message: string; offersNewLayer: boolean; title: string } => {
  const keys =
    overlay && refusal.status === 'layer-hidden' ? OVERLAY_EXCLUDED_KEYS : LAYER_REFUSAL_KEYS[refusal.status];
  const offerMessage = refusal.canPaintNewLayer ? keys.offerMessage : null;
  return {
    message: t(offerMessage ?? keys.message),
    offersNewLayer: offerMessage !== null,
    title: t(keys.title, { name: refusal.layerName }),
  };
};

/** Words a refusal that names its layer: an image that could not be read, or adjustments that could not run. */
export const describeNamedRefusal = (refusal: AdjustmentRefusal | LayerUnreadable, t: TFunction): string =>
  refusal.status === 'unreadable'
    ? t('widgets.layers.actions.unreadable', { name: refusal.layerName })
    : describeAdjustmentRefusal(refusal, t);

/** Explains a refused layer pixel operation; nothing changed in the document, pixels or history. */
export const reportLayerOperation = (
  refusal: LayerOperationRefusal | AdjustmentRefusal | LayerUnreadable,
  reportError: (title: string, message: string) => void,
  t: TFunction
): void =>
  reportError(
    t('widgets.layers.actions.actionFailed'),
    typeof refusal === 'string' ? t(LAYER_OPERATION_KEYS[refusal]) : describeNamedRefusal(refusal, t)
  );

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

/** `busy` stays silent: an in-flight operation already disables the controls that could reach here. */
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

export interface CanvasPreparedEngine {
  readonly document: Pick<CanvasDocumentCapability, 'model'>;
  readonly layers: Pick<CanvasLayerCapability, 'commitPrepared' | 'endStructuralPreview'>;
}

export type PreparedCommitOutcome =
  | StructuralCommitResult
  | { status: 'refused'; refusal: DocumentRefusal }
  | { status: 'unchanged' };

/** Prepares against the current model and commits; open previews end first so the inverse sees the commit. */
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

/** `baseline` restores the previewed state for `before`; null records from the live document. */
export type PreviewCommit = (
  label: string,
  prepare: (model: CanvasDocumentModel, baseline: CanvasLayerPreviewMutation | null) => PrepareEditResult
) => PreparedCommitOutcome;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** `source` narrowed to `shape`'s fields, or undefined if one is missing: the model refuses mismatched `before`. */
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
  /** What the open preview session replaced; null before the first preview and once the engine ends it. */
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
      // No session yet, or the engine ended it: the next session starts from the document as it is now.
      sessionRef.current = engine.layers.beginStructuralPreview() ?? (session?.isActive() ? session : null);
      return sessionRef.current?.apply(action) ?? false;
    },
    [engine]
  );

  const commit = useCallback<PreviewCommit>(
    (label, prepare) => {
      const session = sessionRef.current;
      sessionRef.current = null;
      // The engine ended the gesture: a replay, a newer preview, or any edit or committed-document read from elsewhere
      // (a stroke, a selection edit, a staged result) restored its baseline. Its edit then commits like any prepared
      // edit, silently on refusal: usually the refusal is the interrupting change's own consequence (an undone
      // conversion, lock or removal), and otherwise the previews have already reverted on screen, so the control shows
      // the document as it is.
      const sessionEnded = session !== null && !session.isActive();
      // Without a live session of its own, the edit is prepared from the committed document: a preview another gesture
      // holds ends first, as for any one-shot commit, or its values would leak into the edit and its revision would
      // turn stale when the commit ends it.
      if (engine && !session?.isActive()) {
        engine.layers.endStructuralPreview();
      }
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
      if (!(sessionEnded && outcome.status === 'refused')) {
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
