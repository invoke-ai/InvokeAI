import type {
  CanvasControlLayerContract,
  CanvasLayerContract,
  CanvasRegionalGuidanceLayerContract,
} from '@workbench/canvas-engine/api';

/**
 * The single source of truth for "does this layer carry pixels the generation
 * pipeline will actually use".
 *
 * Both the composite planner (which decides what reaches the backend) and the
 * invoke readiness checks (which decide what the user is told) gate on these
 * predicates. They used to be two hand-maintained copies; a desync there means
 * readiness silently disagrees with what the graph does — a layer reported ready
 * but dropped from the composite, or vice versa.
 *
 * Depends only on layer contracts, so the pure planner can import it without
 * pulling in models, graph, or React.
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
