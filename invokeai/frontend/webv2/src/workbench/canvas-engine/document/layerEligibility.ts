import type { CanvasLayerContract, CanvasNodeContract } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';

import { isGroupNode } from './documentTree';

/** Whether the layer takes part in rendering, generation and export. Display hiding never changes this. */
export const isLayerContributing = (layer: CanvasLayerContract): boolean => layer.isEnabled;

/** Whether the layer accepts document edits that move or repaint it. */
export const isLayerEditable = (layer: CanvasLayerContract): boolean => layer.isEnabled && !layer.isLocked;

/** Raster transparency lock: pixels may change, alpha may not. Brushes composite source-atop; erasing is refused. */
export const isLayerTransparencyLocked = (layer: CanvasLayerContract): boolean =>
  layer.type === 'raster' && layer.isTransparencyLocked === true;

/** Whether a stroke may land directly on the layer's pixels or mask. Control layers edit through a pixel-edit transaction. */
export const isLayerPaintable = (layer: CanvasLayerContract): boolean =>
  isLayerEditable(layer) && layer.type !== 'control';

/** Overlay layers show where an effect applies, so they can be hidden on screen without leaving generation. */
export type HideableLayer = Extract<CanvasLayerContract, { type: 'control' | 'inpaint_mask' | 'regional_guidance' }>;

export const isHideableLayer = (layer: CanvasLayerContract): layer is HideableLayer =>
  layer.type === 'control' || layer.type === 'inpaint_mask' || layer.type === 'regional_guidance';

/** Display only: generation and export never consult it. Groups are hideable in overlay stacks. */
export const isLayerHidden = (layer: CanvasLayerContract): boolean => isHideableLayer(layer) && layer.isHidden === true;

export const isNodeHidden = (node: CanvasNodeContract): boolean =>
  isGroupNode(node) ? node.isHidden === true : isLayerHidden(node);

/** Whether the layer holds pixels rather than a parametric source. */
export const isPixelBackedLayer = (layer: CanvasLayerContract): boolean =>
  (layer.type === 'raster' || layer.type === 'control') &&
  (layer.source.type === 'image' || layer.source.type === 'paint');

/** Enabled, unlocked raster pixels the engine can merge into or delete: masks, control, and parametric sources are not. */
export const isMergeableRasterLayer = (layer: CanvasLayerContract): boolean =>
  isLayerEditable(layer) && layer.type === 'raster' && isPixelBackedLayer(layer);

/** {@link isLayerEditable} with the leaf's ancestors applied: a disabled or locked group gates every descendant. */
export const isLeafEditable = (leaf: SemanticLeaf | null | undefined): leaf is SemanticLeaf =>
  !!leaf && leaf.contributionEnabled && !leaf.effectiveLocked;

/** {@link isLayerPaintable} with the leaf's ancestors applied. */
export const isLeafPaintable = (leaf: SemanticLeaf | null | undefined): leaf is SemanticLeaf =>
  isLeafEditable(leaf) && leaf.layer.type !== 'control';

/** {@link isMergeableRasterLayer} with the leaf's ancestors applied. */
export const isMergeableRasterLeaf = (leaf: SemanticLeaf | null | undefined): leaf is SemanticLeaf =>
  isLeafEditable(leaf) && leaf.layer.type === 'raster' && isPixelBackedLayer(leaf.layer);
