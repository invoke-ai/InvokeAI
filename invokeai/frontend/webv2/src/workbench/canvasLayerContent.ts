import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

/**
 * Shared pixel-content predicates for composite planning and invocation readiness; depends only on layer
 * contracts.
 */

/** True when a control layer holds an image source or committed paint pixels. */
export const hasControlLayerContent = (layer: CanvasControlLayerContract): boolean => {
  if (layer.source.type === 'image') {
    return true;
  }
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

/** True when a regional-guidance layer holds a persisted (non-empty) mask. */
export const hasRegionalGuidanceMaskContent = (layer: CanvasRegionalGuidanceLayerContract): boolean =>
  layer.mask.bitmap !== null;

/** Narrows to the control layers that reach the generation pipeline: enabled and content-bearing. */
export const isCompositableControlLayer = (layer: CanvasLayerContract): layer is CanvasControlLayerContract =>
  layer.type === 'control' && layer.isEnabled && hasControlLayerContent(layer);

/** Narrows to the regional-guidance layers that reach the generation pipeline: enabled and mask-bearing. */
export const isCompositableRegionalGuidanceLayer = (
  layer: CanvasLayerContract
): layer is CanvasRegionalGuidanceLayerContract =>
  layer.type === 'regional_guidance' && layer.isEnabled && hasRegionalGuidanceMaskContent(layer);
