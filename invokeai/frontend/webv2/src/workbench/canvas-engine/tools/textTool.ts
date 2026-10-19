/**
 * Text clicks edit the topmost eligible hit layer or open creation at the point. Measured cache bounds fall back
 * to estimates, with inverse-transform hit testing.
 *
 * An open session consumes the next canvas press after committing live portal content; blur and Mod+Enter also
 * commit. Escape cancels through the portal or engine ladder. Temporary switches preserve sessions; real switches
 * cancel. Creation adds no layer until its single commit.
 */

import type {
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasLayerSourceContract,
} from '@workbench/canvas-engine/contracts';
import type { Vec2 } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves } from '@workbench/canvas-engine/document-model/documentModel';
import { isLeafEditable } from '@workbench/canvas-engine/document/layerEligibility';
import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
import { estimateTextExtent } from '@workbench/canvas-engine/render/rasterizers/textRasterizer';

import type { Tool, ToolContext } from './tool';

import { layerMatrix } from './moveHitTest';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

type TextSource = Extract<CanvasLayerSourceContract, { type: 'text' }>;
/** A text-sourced raster layer. */
type TextLayer = Extract<CanvasLayerContract, { type: 'raster' }> & { source: TextSource };

const isTextLayer = (layer: CanvasLayerContract): layer is TextLayer =>
  layer.type === 'raster' && layer.source.type === 'text';

/** Hit-test measured cache bounds when available, otherwise estimate before first rasterization. */
const textLayerSize = (layer: TextLayer, ctx: ToolContext): { width: number; height: number } => {
  const cache = ctx.layers.get(layer.id);
  if (cache) {
    return { height: cache.surface.height, width: cache.surface.width };
  }
  return estimateTextExtent(layer.source);
};

/** The top-most editable text layer whose rendered block contains `point` (document space), or `null`. */
const topTextLayerAt = (doc: CanvasDocumentContractV3, point: Vec2, ctx: ToolContext): TextLayer | null => {
  for (const leaf of compileDocumentLeaves(doc)) {
    const layer = leaf.layer;
    if (!isTextLayer(layer) || !isLeafEditable(leaf)) {
      continue;
    }
    const inverse = invert(layerMatrix(layer.transform));
    if (!inverse) {
      continue;
    }
    const local = applyToPoint(inverse, point);
    const size = textLayerSize(layer, ctx);
    if (local.x >= 0 && local.x <= size.width && local.y >= 0 && local.y <= size.height) {
      return layer;
    }
  }
  return null;
};

/** Creates a fresh text tool. It holds no gesture state (a click opens a session and returns). */
export const createTextTool = (): Tool => ({
  cursor: () => 'text',
  id: 'text',
  onDeactivate: (ctx, opts) => {
    if (opts?.temporary) {
      // Preserve text sessions across modifier holds for later reactivation.
      return;
    }
    // Real switches cancel remaining sessions after any portal-blur commit.
    ctx.cancelTextEdit?.();
  },
  onKeyCommand: (ctx, command) => {
    // Direct harness cancellation is a backstop; production Escape uses the engine ladder. Apply needs portal
    // content and does nothing here.
    if (command === 'cancel') {
      ctx.cancelTextEdit?.();
    }
  },
  onPointerDown: (ctx, input) => {
    if ((input.buttons & PRIMARY_BUTTON) === 0) {
      return;
    }
    // Defensively reject a second session if a harness bypasses the pipeline's commit-and-consume hook.
    if (ctx.stores.textEditSession.get()) {
      return;
    }
    const doc = ctx.getDocument();
    if (!doc) {
      return;
    }
    const hit = topTextLayerAt(doc, input.documentPoint, ctx);
    if (hit) {
      ctx.openTextEdit?.(hit.id);
    } else {
      ctx.openTextCreate?.(input.documentPoint);
    }
  },
});
