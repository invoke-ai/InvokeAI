import type {
  CanvasDocumentContractV3,
  CanvasLayerEditRefusal,
  CanvasNoTargetRefusal,
  CanvasNodeContract,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { createItemId } from '@platform/browser/randomUuid';
import { useMountEffect } from '@platform/react/useMountEffect';
import { captureAccountScope } from '@platform/state/accountLifecycle';
import { ACTION_TOAST_DURATION_MS, createActionToast, toaster } from '@platform/ui/toaster';
import { getDocumentIndex, isNodeHidden, isOverlayStack } from '@workbench/canvas-engine/api';
import { nextLayerName } from '@workbench/canvasProjectMutations';
import { useNotify } from '@workbench/useNotify';
import {
  commitPreparedEdit,
  describeAdjustmentRefusal,
  describeLayerEditRefusal,
  reportPreparedCommit,
  reportStructuralCommit,
  type CanvasPreparedEngine,
  type PreparedCommitOutcome,
} from '@workbench/widgets/canvas/useStructuralCommit';
import { createEmptyPaintLayer } from '@workbench/widgets/layers/layerOps';
import { useTranslation } from 'react-i18next';

const NO_TARGET_KEYS: Record<CanvasNoTargetRefusal['status'], { message: string; title: string }> = {
  'cutout-no-target': {
    message: 'widgets.canvas.editRefusal.noCutoutTargetMessage',
    title: 'widgets.canvas.editRefusal.noCutoutTarget',
  },
  'erase-no-target': {
    message: 'widgets.canvas.editRefusal.noEraseTargetMessage',
    title: 'widgets.canvas.editRefusal.noEraseTarget',
  },
  'selection-erase-no-target': {
    message: 'widgets.canvas.editRefusal.noSelectionEraseTargetMessage',
    title: 'widgets.canvas.editRefusal.noSelectionEraseTarget',
  },
};

/**
 * Where the new layer goes so the refusal's cause does not reach it: above the refused layer, or above its outermost
 * group that hides or locks it, since a layer inside that group would be refused the same way. Null (the top) once the
 * refused layer is gone.
 */
export const paintOfferAnchorId = (
  document: Pick<CanvasDocumentContractV3, 'stacks'>,
  refusal: Pick<CanvasLayerEditRefusal, 'layerId' | 'status'>
): string | null => {
  const index = getDocumentIndex(document);
  const entry = index.byId.get(refusal.layerId);
  if (!entry) {
    return null;
  }
  const causes = (node: CanvasNodeContract): boolean =>
    refusal.status === 'layer-locked' ? node.isLocked : !node.isEnabled || isNodeHidden(node);
  return (
    entry.path.find((id) => {
      const ancestor = index.byId.get(id)?.node;
      return ancestor !== undefined && causes(ancestor);
    }) ?? refusal.layerId
  );
};

/** Adds an empty paint layer above the refused layer (see {@link paintOfferAnchorId}) and selects it. */
const paintOnNewLayer = (
  engine: CanvasPreparedEngine,
  refusal: CanvasLayerEditRefusal,
  label: string
): PreparedCommitOutcome =>
  commitPreparedEdit(engine, label, (model) =>
    model.prepare({
      aboveId: paintOfferAnchorId(model.document, refusal),
      nodes: [
        createEmptyPaintLayer(nextLayerName(getDocumentIndex(model.document).nodes.map(({ node }) => node.name))),
      ],
      type: 'insert',
    })
  );

/**
 * Explains edits the engine refused without a caller to report to: a stroke too large to undo, a press on a layer
 * the tool cannot edit (hidden, hidden on the canvas, locked, still loading, one the eraser cannot erase, or a mask
 * the gradient cannot go on), or an erase or shape cutout with no layer selected. A brush press on a locked or hidden
 * image offers a new paint layer above it rather than making one unasked; that offer holds a closure, so it is a
 * toast only, closed with this canvas or the account. Layer refusals are toasts only, one per layer and reason: a
 * repeated press (every eraser press on a transparency-locked layer, say) replaces its toast instead of stacking
 * another, and taking one offer closes the other offers. An erase or cutout with nothing to act on keeps a single
 * toast the same way. Keyed by engine.
 */
export const CanvasEditRefusalNotices = ({
  engine,
}: {
  engine: Pick<CanvasEngineHandle, 'tools'> & CanvasPreparedEngine;
}) => {
  const notify = useNotify();
  const { t } = useTranslation();

  useMountEffect(() => {
    const mounted = new AbortController();
    const scope = createItemId('canvas-edit-refusal');
    const offers = new Set<string>();
    const unsubscribe = engine.tools.onEditRefused((refusal) => {
      if (typeof refusal === 'string') {
        if (refusal === 'busy') {
          // Elsewhere `busy` stays silent because its controls are disabled; a stroke can still meet it mid-gesture.
          notify.error(t('widgets.canvas.structural.failed'), t('widgets.canvas.structural.busy'));
        } else {
          reportStructuralCommit({ status: refusal }, notify.error, t);
        }
        return;
      }
      if (refusal.status === 'adjustment-refused') {
        notify.error(t('widgets.canvas.structural.failed'), describeAdjustmentRefusal(refusal, t));
        return;
      }
      const signal = AbortSignal.any([captureAccountScope().signal, mounted.signal]);
      if (!('layerId' in refusal)) {
        const keys = NO_TARGET_KEYS[refusal.status];
        createActionToast({
          actions: [],
          description: t(keys.message),
          id: `${scope}:${refusal.status}`,
          signal,
          title: t(keys.title),
          type: 'info',
        });
        return;
      }
      const document = engine.document.model()?.document;
      const stack = document ? getDocumentIndex(document).byId.get(refusal.layerId)?.stack : undefined;
      const { message, offersNewLayer, title } = describeLayerEditRefusal(
        refusal,
        t,
        stack !== undefined && isOverlayStack(stack)
      );
      // One toast per layer and reason: pressing again on the same layer replaces it rather than stacking another.
      const id = `${scope}:${refusal.status}:${refusal.layerId}`;
      if (!offersNewLayer) {
        createActionToast({ actions: [], description: message, id, signal, title, type: 'info' });
        return;
      }
      offers.add(id);
      createActionToast({
        actions: [
          {
            label: t('widgets.canvas.editRefusal.paintOnNewLayer'),
            onClick: () => {
              for (const other of offers) {
                if (other !== id) {
                  toaster.dismiss(other);
                }
              }
              reportPreparedCommit(
                paintOnNewLayer(engine, refusal, t('widgets.layers.actions.addRasterLayer')),
                notify.error,
                t
              );
            },
          },
        ],
        description: message,
        duration: ACTION_TOAST_DURATION_MS,
        id,
        onStatusChange: ({ status }) => {
          if (status === 'unmounted') {
            offers.delete(id);
          }
        },
        signal,
        title,
        type: 'info',
      });
    });
    return () => {
      unsubscribe();
      mounted.abort();
    };
  });
  return null;
};
