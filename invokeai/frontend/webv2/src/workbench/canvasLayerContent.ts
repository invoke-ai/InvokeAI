import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

/**
 * Shared pixel-content predicates for composite planning, invocation readiness and control-image state; depends only
 * on layer contracts.
 */

/** True when a control layer holds an image source or committed paint pixels. */
export const hasControlLayerContent = (layer: CanvasControlLayerContract): boolean => {
  if (layer.source.type === 'image') {
    return true;
  }
  return layer.source.type === 'paint' && layer.source.bitmap !== null;
};

/** `drawn`: paint (or another non-image source) with pixels; `empty`: nothing to detect or use yet. */
export type ControlImageState = 'empty' | 'image' | 'drawn';

/**
 * What a control layer holds to detect from or use as-is. `sourceHasPixels` is the engine's answer for strokes the
 * document has not persisted yet.
 */
export const getControlImageState = (
  layer: Pick<CanvasControlLayerContract, 'original' | 'source'>,
  sourceHasPixels: boolean
): ControlImageState => {
  const input = layer.original ?? layer.source;
  if (input.type === 'image') {
    return 'image';
  }
  if (input.type === 'paint' && input.bitmap !== null) {
    return 'drawn';
  }
  // Unpersisted strokes (or a parametric source) only the engine's cache knows about.
  return !layer.original && sourceHasPixels ? 'drawn' : 'empty';
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
