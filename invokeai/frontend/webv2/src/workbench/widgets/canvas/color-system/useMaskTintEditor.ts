import type {
  CanvasDocumentContractV3,
  CanvasDocumentModel,
  CanvasLayerContract,
  CanvasLayerPreviewMutation,
  CanvasMaskFillContract,
  PrepareEditResult,
} from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { getDocumentLayer } from '@workbench/canvas-engine/api';
import { baselineConfig, useStructuralPreview } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  clearMaskTintTarget,
  holdPendingMaskTint,
  releasePendingMaskTint,
  useMaskTintTargetLayerId,
} from './maskTintTarget';

type MaskTintEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'layers'>;
type MaskLayer = Extract<CanvasLayerContract, { type: 'inpaint_mask' | 'regional_guidance' }>;

/** Key repeats and drag releases coalesce into one entry after this idle gap. */
const COMMIT_SETTLE_MS = 400;

export interface MaskTintEditor {
  layerId: string;
  /** The armed mask's current tint. */
  color: string;
  /** Live, un-recorded preview during a drag. */
  preview: (hex: string) => void;
  /** Applies the color and records the gesture's ONE history entry once it settles. */
  commit: (hex: string) => void;
}

/**
 * Capture the layer id, latest color, and timer together so commits survive rerenders, disarming, and unmount. The
 * engine's preview session holds the fill the gesture started from.
 */
interface TintGesture {
  layerId: string;
  latestHex: string;
  timer: ReturnType<typeof setTimeout> | null;
}

export const isMaskLayer = (layer: CanvasLayerContract | null): layer is MaskLayer =>
  layer?.type === 'inpaint_mask' || layer?.type === 'regional_guidance';

const configFor = (layer: MaskLayer, fill: CanvasMaskFillContract) =>
  layer.type === 'inpaint_mask'
    ? ({ layerType: 'inpaint_mask', mask: { fill } } as const)
    : ({ layerType: 'regional_guidance', mask: { fill } } as const);

/** The history label every tint edit records under, whichever control or pick made it. */
export const MASK_TINT_EDIT_LABEL_KEY = 'widgets.layers.maskFill.fill';

/** The armed tint target's layer while it is still the selected mask; anything else disarms it. */
export const armedMaskLayer = (document: CanvasDocumentContractV3, armedLayerId: string | null): MaskLayer | null => {
  if (!armedLayerId || document.selectedLayerId !== armedLayerId) {
    return null;
  }
  const node = getDocumentLayer(document, armedLayerId);
  return node && isMaskLayer(node) ? node : null;
};

/** The un-recorded live preview of `layer` tinted `hex`. */
export const maskTintPreview = (layer: MaskLayer, hex: string): CanvasLayerPreviewMutation => ({
  config: configFor(layer, { ...layer.mask.fill, color: hex }),
  id: layer.id,
  type: 'updateCanvasLayerConfig',
});

/** The recorded tint edit for the live layer, its `before` taken from what a preview gesture replaced. */
export const prepareMaskTintEdit = (
  model: CanvasDocumentModel,
  baseline: CanvasLayerPreviewMutation | null,
  layerId: string,
  hex: string
): PrepareEditResult => {
  const live = model.getLayer(layerId);
  if (!isMaskLayer(live)) {
    return { ids: [layerId], status: 'missing' };
  }
  const config = configFor(live, { ...live.mask.fill, color: hex });
  return model.prepare({ before: baselineConfig(baseline, config), config, id: live.id, type: 'patch-config' });
};

/**
 * Clear invalid/unselected mask targets and return to pair editing. Settle each pending preview into exactly one
 * history entry on idle, disarm, or unmount.
 */
export const useMaskTintEditor = (engine: MaskTintEngine | null): MaskTintEditor | null => {
  const { t } = useTranslation();
  const { commit: commitPrepared, preview: previewStructural } = useStructuralPreview(engine);
  const armedLayerId = useMaskTintTargetLayerId();
  const layer = useActiveProjectSelector((project) => armedMaskLayer(project.canvas.document, armedLayerId));

  const gestureRef = useRef<TintGesture | null>(null);
  const settleRef = useRef<() => void>(() => undefined);
  // One stable identity for the shared pending-gesture slot, always running the latest settle.
  const [settleLatest] = useState(() => () => settleRef.current());
  const settle = useCallback(() => {
    const gesture = gestureRef.current;
    gestureRef.current = null;
    if (!gesture) {
      return;
    }
    releasePendingMaskTint(settleLatest);
    if (gesture.timer !== null) {
      clearTimeout(gesture.timer);
    }
    // A gesture ending on the color it started from prepares nothing, which drops its previews.
    commitPrepared(t(MASK_TINT_EDIT_LABEL_KEY), (model, baseline) =>
      prepareMaskTintEdit(model, baseline, gesture.layerId, gesture.latestHex)
    );
  }, [commitPrepared, settleLatest, t]);
  useEffect(() => {
    settleRef.current = settle;
  }, [settle]);

  useEffect(() => {
    if (armedLayerId && !layer) {
      clearMaskTintTarget();
    }
  }, [armedLayerId, layer]);
  // Disarm and unmount both flush the pending gesture; the gesture carries its
  // own layer, so this works even after the selection has moved on.
  useEffect(() => {
    if (!armedLayerId) {
      settleRef.current();
    }
    return () => settleRef.current();
  }, [armedLayerId]);

  const preview = useCallback(
    (hex: string) => {
      if (!layer) {
        return;
      }
      if (!previewStructural(maskTintPreview(layer, hex))) {
        return;
      }
      if (gestureRef.current === null) {
        gestureRef.current = { latestHex: hex, layerId: layer.id, timer: null };
        holdPendingMaskTint(settleLatest);
      } else {
        gestureRef.current.latestHex = hex;
      }
    },
    [previewStructural, layer, settleLatest]
  );
  const commit = useCallback(
    (hex: string) => {
      preview(hex);
      const gesture = gestureRef.current;
      if (!gesture) {
        return;
      }
      if (gesture.timer !== null) {
        clearTimeout(gesture.timer);
      }
      gesture.timer = setTimeout(() => settleRef.current(), COMMIT_SETTLE_MS);
    },
    [preview]
  );

  return useMemo(
    () => (layer ? { color: layer.mask.fill.color, commit, layerId: layer.id, preview } : null),
    [commit, layer, preview]
  );
};
