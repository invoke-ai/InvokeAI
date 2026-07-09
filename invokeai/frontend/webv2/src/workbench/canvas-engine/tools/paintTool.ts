/**
 * The shared machinery behind the brush and eraser tools. Both are the same
 * gesture — resolve a paintable target layer on pointer-down (auto-creating a
 * fresh paint layer when the selection can't be painted into), drive a
 * {@link StrokeSession} across the move batches, and commit or cancel on release —
 * differing only in their blend (`source-over` fill vs `destination-out`) and in
 * which options store they read. `createBrushTool` / `createEraserTool`
 * (in `brushTool.ts` / `eraserTool.ts`) are thin wrappers over
 * {@link createPaintTool}.
 *
 * This is the ONE place a painting tool is allowed to dispatch, and only ever
 * the single gesture-start `addCanvasLayer`. Pointer-move never dispatches.
 *
 * Zero React, zero import-time side effects.
 */

import type { PointerInput } from '@workbench/canvas-engine/types';
import type { CanvasLayerContract, CanvasRasterLayerContractV2 } from '@workbench/types';

import type { Tool, ToolContext } from './tool';

import { createStrokeSession, type StrokeSession } from './strokeSession';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** The resolved config for one gesture, derived from the tool's options store. */
export interface PaintToolSpec {
  id: 'brush' | 'eraser';
  composite: 'source-over' | 'destination-out';
  /** Reads the current size (document units) from the options store. */
  size(ctx: ToolContext): number;
  /** Reads the current per-stroke opacity from the options store. */
  opacity(ctx: ToolContext): number;
  /** The fill color; `null` for the eraser (shape only). */
  color(ctx: ToolContext): string;
  /** Freehand thinning for this gesture; 0 disables pressure sensitivity. */
  thinning(ctx: ToolContext): number;
}

/** Colour brush strokes paint into a MASK cache: an opaque stencil (only alpha matters). */
const MASK_STROKE_COLOR = '#ffffff';

/** The resolved paint target for a gesture. `createdLayer` is set only when auto-created. */
interface PaintTarget {
  layerId: string;
  /** When the gesture auto-created its layer, the created contract + insert index (for history). */
  createdLayer?: { layer: CanvasLayerContract; index: number };
  /**
   * Overrides the brush colour for this gesture (mask targets paint an opaque
   * stencil — the stored RGB is irrelevant, the compositor colorizes by alpha).
   * Absent ⇒ the tool's own colour.
   */
  color?: string;
  /**
   * True when the target is a transparency-LOCKED raster paint layer. The brush
   * then composites `source-atop` (colour only on existing pixels); the eraser is
   * refused (erasing would change the locked alpha). Never set for mask targets.
   */
  transparencyLocked?: boolean;
  /**
   * True for mask targets (inpaint / regional guidance). A mask is an opaque
   * alpha stencil, so the stroke is forced to opacity 1 regardless of the brush's
   * opacity slider — a 50%-opacity brush would otherwise land alpha ~128 and
   * silently attenuate the mask (a ~50% denoise, invisible in the tinted overlay).
   */
  forceOpaque?: boolean;
}

/** True for a mask-bearing layer (inpaint mask / regional guidance) — a paintable alpha stencil. */
const isMaskLayer = (layer: CanvasLayerContract): boolean =>
  layer.type === 'inpaint_mask' || layer.type === 'regional_guidance';

/** Resolves (or auto-creates) the paint target for a gesture, or `null` to no-op. */
const resolveTarget = (ctx: ToolContext): PaintTarget | null => {
  const doc = ctx.getDocument();
  if (!doc) {
    return null;
  }
  const selected = doc.selectedLayerId ? doc.layers.find((layer) => layer.id === doc.selectedLayerId) : undefined;

  if (selected && selected.type === 'raster' && selected.source.type === 'paint') {
    // The selection is a paint layer: paint into it, unless it's locked/disabled
    // (a no-op — don't silently spawn a new layer over the user's locked target).
    if (selected.isLocked || !selected.isEnabled) {
      return null;
    }
    // The cache (if any) keeps its current content extent; the stroke grows it.
    return { layerId: selected.id, transparencyLocked: selected.isTransparencyLocked === true };
  }

  if (selected && isMaskLayer(selected)) {
    // The selection is a mask: paint the stroke into its alpha stencil cache
    // (brush adds coverage, eraser removes — the shared stroke session handles
    // both via its composite op). Never auto-create a paint layer here. A
    // locked/hidden mask refuses the stroke (a no-op, not a spawn).
    if (selected.isLocked || !selected.isEnabled) {
      return null;
    }
    return { color: MASK_STROKE_COLOR, forceOpaque: true, layerId: selected.id };
  }

  // Selection is an image/other raster, another layer type, or nothing: create a
  // fresh paint layer (inserted on top and selected by the reducer) and paint
  // into it. This is the single allowed gesture-start dispatch.
  const layerId = ctx.createLayerId();
  const layer: CanvasRasterLayerContractV2 = {
    blendMode: 'normal',
    id: layerId,
    isEnabled: true,
    isLocked: false,
    name: `Layer ${doc.layers.length + 1}`,
    opacity: 1,
    source: { bitmap: null, type: 'paint' },
    transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
    type: 'raster',
  };
  // Auto-create inserts at the top (index 0); the reducer selects it.
  ctx.dispatch({ layer, type: 'addCanvasLayer' });

  // A brand-new empty paint layer: create a zero-rect cache marked fresh so the
  // async rasterize pass doesn't clobber the stroke mid-gesture. The first stroke
  // grows it from empty to the stroke's content bounds.
  const entry = ctx.layers.getOrCreateRect(layerId, { height: 0, width: 0, x: 0, y: 0 });
  entry.stale = false;
  return { createdLayer: { index: 0, layer }, layerId };
};

/** Creates a brush-family tool from its per-gesture {@link PaintToolSpec}. */
export const createPaintTool = (spec: PaintToolSpec): Tool => {
  let session: StrokeSession | null = null;

  const cursorRadiusDoc = (ctx: ToolContext): number => spec.size(ctx) / 2;

  const updateCursorRing = (ctx: ToolContext, input: PointerInput): void => {
    ctx.setOverlayCursor({ point: input.documentPoint, radiusDoc: cursorRadiusDoc(ctx) });
    ctx.invalidate({ overlay: true });
  };

  const endSession = (): void => {
    session = null;
  };

  return {
    cursor: () => 'crosshair',
    id: spec.id,
    onDeactivate: (ctx) => {
      if (session) {
        session.cancel();
        endSession();
      }
      ctx.setOverlayCursor(null);
      ctx.invalidate({ overlay: true });
    },
    onPointerCancel: () => {
      if (session) {
        session.cancel();
        endSession();
      }
    },
    onPointerDown: (ctx, input) => {
      if (session || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const target = resolveTarget(ctx);
      updateCursorRing(ctx, input);
      if (!target) {
        return;
      }
      // Transparency lock: the eraser is refused (it would alter the locked alpha);
      // the brush switches to `source-atop` so colour lands only on existing pixels.
      if (target.transparencyLocked && spec.id === 'eraser') {
        return;
      }
      const composite = target.transparencyLocked && spec.id === 'brush' ? 'source-atop' : spec.composite;
      session = createStrokeSession({
        // Resolve the selection clip ONCE per gesture: when a selection exists the
        // stroke is masked to it; with none the field is null and the hot path is
        // untouched (no per-point mask lookup).
        clipMask: ctx.getSelectionMask?.() ?? null,
        color: target.color ?? spec.color(ctx),
        composite,
        createdLayer: target.createdLayer ?? null,
        ctx,
        layerId: target.layerId,
        // Mask strokes are forced opaque (an alpha stencil is all-or-nothing); a
        // brush-opacity mask stroke would silently attenuate the denoise strength.
        opacity: target.forceOpaque ? 1 : spec.opacity(ctx),
        size: spec.size(ctx),
        thinning: spec.thinning(ctx),
        tool: spec.id,
      });
      session.addPoints([input]);
    },
    onPointerMove: (ctx, input, batch) => {
      updateCursorRing(ctx, input);
      if (session) {
        session.addPoints(batch);
      }
    },
    onPointerUp: (ctx, input) => {
      updateCursorRing(ctx, input);
      if (session) {
        session.commit();
        endSession();
      }
    },
  };
};
