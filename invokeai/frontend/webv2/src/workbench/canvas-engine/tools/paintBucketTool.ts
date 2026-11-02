/**
 * Paint bucket: a click fills the region of similar color under the pointer on the selected layer (or erases it), and
 * holding and dragging sideways scrubs the tolerance with the region redrawn live; release records the last drawn
 * region as one step and keeps the tolerance it landed on. The target follows the brush's rules (`resolvePaintTarget`).
 *
 * A press composites its sample domain once and hands it to the flood fill worker, which keeps the press's fill level
 * map, so each tolerance change is one cheap region request. Every answer newer than the one on the canvas redraws the
 * region into the layer cache through one coverage edit. Release waits for the last request's answer, so the
 * committed pixels are the previewed ones; until then the engine counts the press as part of its gesture.
 */

import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/contracts';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { PaintBucketOptions } from '@workbench/canvas-engine/engineStores';
import type { CompositeOptions } from '@workbench/canvas-engine/render/compositor';
import type { FillSample } from '@workbench/canvas-engine/render/fill/fillLevelMap';
import type { FloodFillAnswer } from '@workbench/canvas-engine/render/fill/floodFillService';
import type { FillRegion, FillRegionParams } from '@workbench/canvas-engine/render/fill/regionMorphology';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PlacedSurface, PointerInput, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { paintsInPlace } from '@workbench/canvas-engine/document/layerEligibility';
import { isMaskLayer } from '@workbench/canvas-engine/document/sources';
import { MAX_FILL_TOLERANCE } from '@workbench/canvas-engine/engineStores';
import { identity, invert, multiply, translate } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { compositeDocument, prepareComposite } from '@workbench/canvas-engine/render/compositor';
import { exceedsFillPixelCap, resolveFillDomain } from '@workbench/canvas-engine/render/fill/fillDomain';

import type { Tool, ToolContext } from './tool';

import { clipCoverageRect, createCoverageEdit, type CoverageEdit, type CoverageOperation } from './coverageEdit';
import { MASK_PAINT_COLOR } from './paintConstants';
import {
  paintTargetMatrix,
  predictPaintRefusal,
  refusePaintTarget,
  resolvePaintTarget,
  type PaintTarget,
} from './paintTool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Screen-space distance (CSS px) the pointer must travel before a press scrubs the tolerance. */
export const PAINT_BUCKET_DRAG_THRESHOLD_PX = 3;

/** Horizontal CSS pixels of drag per tolerance percent. */
export const PAINT_BUCKET_PX_PER_TOLERANCE = 4;

/** "All layers" is what the eye sees of the raster and control stacks: no mask or region tints. */
const SAMPLED_STACKS: CompositeOptions['showOverlayStacks'] = {
  control: true,
  inpaint_mask: false,
  regional_guidance: false,
};

const NO_PIXELS: Rect = { height: 0, width: 0, x: 0, y: 0 };

/** A sample's peak footprint: the composite surface and the pixels read back from it are alive together. */
const SAMPLE_PEAK_BYTES_PER_PIXEL = 8;

/** The region on the canvas: where it was drawn (document space) and its coverage over `bounds` (sample space). */
interface DrawnRegion {
  readonly docRect: Rect;
  readonly bounds: Rect;
  readonly coverage: Uint8Array;
}

interface Press {
  readonly id: number;
  readonly ctx: ToolContext;
  readonly target: PaintTarget;
  readonly coverage: CoverageEdit;
  readonly erase: boolean;
  /** The sample's document rect; region bounds are relative to its origin. */
  readonly domain: Rect;
  readonly clipMask: PlacedSurface | null;
  readonly clipRect: Rect | null;
  /** What the coverage paints with, and at what alpha; masks paint opaque mask color with hard edges. */
  readonly color: string;
  readonly alpha: number;
  readonly hard: boolean;
  readonly params: Omit<FillRegionParams, 'tolerance'>;
  readonly startScreen: Vec2;
  readonly pressTolerance: number;
  tolerance: number;
  scrubbing: boolean;
  released: boolean;
  /** Region requests not answered yet; release commits once none are left. */
  outstanding: number;
  /** The sequence of the answer drawn on the canvas, so an older one arriving late is not drawn over it. */
  drawnSequence: number;
  /** The last region drawn, kept to tell whether the selection's shape left any of it. */
  drawn: DrawnRegion | null;
  /** Reused for each answer's coverage; grows to the largest region bounds so far, with its readback buffer. */
  surface: RasterSurface | null;
  image: ImageData | null;
}

const selectedLeaf = (doc: CanvasDocumentContractV3 | null): SemanticLeaf | null =>
  doc?.selectedLayerId ? lookupDocumentLeaf(doc, doc.selectedLayerId) : null;

/**
 * How a press composites its sample: the display's adjusted, derived and group pixels with the engine's live
 * transform overrides (a materialized image or control layer draws its baked pixels at identity), without checkerboard,
 * staged result, mask or region tints, unsmoothed. One preparation serves both the content bounds and the composite.
 */
interface SamplePlan {
  readonly options: CompositeOptions;
  readonly contentBounds: Rect | null;
}

/** Plans the sample of everything visible, or of `isolationLayerId` alone. */
const planSample = (ctx: ToolContext, doc: CanvasDocumentContractV3, isolationLayerId: string | null): SamplePlan => {
  const base: CompositeOptions = {
    ...ctx.sampleProviders,
    backend: ctx.backend,
    imageSmoothing: false,
    isolationLayerId,
    showOverlayStacks: SAMPLED_STACKS,
    transformOverrides: ctx.transformOverrides ?? null,
  };
  const preparation = prepareComposite(doc, base);
  let contentBounds: Rect | null = null;
  for (let index = 0; index < preparation.leaves.length; index++) {
    const entry = ctx.layers.peek(preparation.leaves[index]!.id);
    if (entry && entry.rect.width > 0 && entry.rect.height > 0) {
      const bounds = transformBounds(preparation.matrices[index]!, entry.rect);
      contentBounds = contentBounds ? union(contentBounds, bounds) : bounds;
    }
  }
  return { contentBounds, options: { ...base, preparation } };
};

/** Composites the planned sample over `domain`. An isolated mask samples its own coverage, not its tinted display. */
const compositeSample = (
  ctx: ToolContext,
  doc: CanvasDocumentContractV3,
  domain: Rect,
  plan: SamplePlan
): FillSample => {
  const surface = ctx.backend.createSurface(domain.width, domain.height);
  try {
    const view = translate(identity(), { x: -domain.x, y: -domain.y });
    const { preparation } = plan.options;
    const isolated = preparation?.leaves.find((leaf) => leaf.id === plan.options.isolationLayerId);
    if (preparation && isolated && isMaskLayer(isolated.layer)) {
      const entry = ctx.layers.peek(isolated.id);
      if (entry && entry.rect.width > 0 && entry.rect.height > 0) {
        const matrix = multiply(view, preparation.matrices[preparation.leaves.indexOf(isolated)]!);
        surface.ctx.imageSmoothingEnabled = false;
        surface.ctx.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
        surface.ctx.drawImage(entry.surface.canvas, entry.rect.x, entry.rect.y);
      }
    } else {
      compositeDocument(surface, doc, ctx.layers, view, plan.options);
    }
    const { data } = surface.ctx.getImageData(0, 0, domain.width, domain.height);
    return { data, height: domain.height, width: domain.width };
  } finally {
    surface.resize(0, 0);
  }
};

/** Whether any drawn coverage lies inside the selection's shape, not merely inside its bounds. */
const reachesSelection = (clipMask: PlacedSurface, domain: Rect, drawn: DrawnRegion): boolean => {
  const { bounds, coverage, docRect } = drawn;
  const { data } = clipMask.surface.ctx.getImageData(
    docRect.x - clipMask.rect.x,
    docRect.y - clipMask.rect.y,
    docRect.width,
    docRect.height
  );
  const offsetX = docRect.x - domain.x - bounds.x;
  const offsetY = docRect.y - domain.y - bounds.y;
  for (let row = 0; row < docRect.height; row++) {
    const coverageRow = (offsetY + row) * bounds.width + offsetX;
    const maskRow = row * docRect.width;
    for (let column = 0; column < docRect.width; column++) {
      if (coverage[coverageRow + column]! > 0 && data[(maskRow + column) * 4 + 3]! > 0) {
        return true;
      }
    }
  }
  return false;
};

/** Creates a fresh paint bucket tool with its own gesture state. */
export const createPaintBucketTool = (): Tool => {
  let press: Press | null = null;
  let nextPressId = 0;

  /** Ends the worker press and the tool's hold on it; the target's pixels are already settled. */
  const endPress = (current: Press): void => {
    if (press === current) {
      press = null;
    }
    const { ctx } = current;
    try {
      ctx.floodFill?.end(current.id);
      current.surface?.resize(0, 0);
      current.surface = null;
      current.image = null;
    } finally {
      ctx.stores.paintBucketScrub.set(null);
      ctx.invalidate({ overlay: true });
      ctx.updateCursor();
    }
  };

  /** Puts the layer's pixels back, removes a layer the press created, and records nothing. */
  const cancelPress = (current: Press): void => {
    if (press === current) {
      press = null;
    }
    try {
      current.coverage.cancel();
    } finally {
      try {
        current.target.cancel();
      } finally {
        endPress(current);
      }
    }
  };

  /**
   * Records the drawn region as one step. Nothing drawn, or a region the selection's shape leaves nothing of, records
   * nothing and removes a layer the press created.
   */
  const finishPress = (current: Press): void => {
    press = null;
    const { clipMask, drawn, target } = current;
    let committed = false;
    try {
      if (!clipMask || !drawn || reachesSelection(clipMask, current.domain, drawn)) {
        committed = current.coverage.commit({
          tool: current.erase ? 'paintBucketErase' : 'paintBucket',
          ...(target.createdLayer ? { createdLayer: target.createdLayer } : {}),
        });
      }
    } finally {
      try {
        if (!committed) {
          current.coverage.cancel();
          target.cancel();
        }
      } finally {
        endPress(current);
      }
    }
  };

  /**
   * Writes the region's coverage, colored, into the top-left `bounds`-sized corner of the press's coverage surface. The
   * surface and its readback buffer only grow, so a scrub reallocates only when a region outgrows every earlier one.
   */
  const coverageSurface = (current: Press, bounds: Rect, coverage: Uint8Array): RasterSurface => {
    const { height, width } = bounds;
    let surface = current.surface;
    if (!surface) {
      surface = current.ctx.backend.createSurface(width, height);
      current.surface = surface;
    } else if (surface.width < width || surface.height < height) {
      surface.resize(Math.max(surface.width, width), Math.max(surface.height, height));
    }
    const draw = surface.ctx;
    let image = current.image;
    if (!image || image.width !== surface.width || image.height !== surface.height) {
      image = draw.createImageData(surface.width, surface.height);
      current.image = image;
    }
    // Only alpha is written; color comes from the fill below.
    const { data } = image;
    const stride = image.width;
    for (let row = 0; row < height; row++) {
      const from = row * width;
      const to = row * stride;
      for (let column = 0; column < width; column++) {
        data[(to + column) * 4 + 3] = coverage[from + column]!;
      }
    }
    draw.setTransform(1, 0, 0, 1, 0, 0);
    draw.putImageData(image, 0, 0, 0, 0, width, height);
    draw.globalCompositeOperation = 'source-in';
    draw.fillStyle = current.color;
    draw.fillRect(0, 0, width, height);
    draw.globalCompositeOperation = 'source-over';
    return surface;
  };

  /** Redraws the layer with `region`; false when the edit could not grow (it has restored and cancelled itself). */
  const drawRegion = (current: Press, region: FillRegion): boolean => {
    const { bounds, coverage } = region;
    const x = current.domain.x + (bounds?.x ?? 0);
    const y = current.domain.y + (bounds?.y ?? 0);
    const docRect = bounds ? clipCoverageRect({ height: bounds.height, width: bounds.width, x, y }, current) : null;
    if (!bounds || !coverage || !docRect) {
      // Nothing to draw: put back whatever the previous answer drew.
      current.drawn = null;
      return current.coverage.apply(NO_PIXELS, () => undefined);
    }
    current.drawn = { bounds, coverage, docRect };
    const surface = coverageSurface(current, bounds, coverage);
    const { height, width } = bounds;
    return current.coverage.apply(docRect, (draw) => {
      draw.imageSmoothingEnabled = !current.hard;
      draw.globalAlpha = current.alpha;
      draw.drawImage(surface.canvas, 0, 0, width, height, x, y, width, height);
    });
  };

  const onAnswer = (current: Press, answer: FloodFillAnswer): void => {
    current.outstanding -= 1;
    if (press !== current) {
      return;
    }
    if (answer.status === 'error' || answer.status === 'ended') {
      // The map is gone (a failed job, a crashed worker, or the engine ended the press): nothing newer can land.
      cancelPress(current);
      return;
    }
    if (answer.status === 'ok' && answer.sequence > current.drawnSequence) {
      current.drawnSequence = answer.sequence;
      if (!drawRegion(current, answer.region)) {
        // The coverage edit restored the pixels and the refusal was reported; a created layer goes too.
        cancelPress(current);
        return;
      }
    }
    if (current.released && current.outstanding === 0) {
      finishPress(current);
    }
  };

  const track = (current: Press, answer: Promise<FloodFillAnswer>): void => {
    current.outstanding += 1;
    // A failure while drawing an answer puts the layer back; the press has nothing left to report it to.
    void answer
      .then((settled) => onAnswer(current, settled))
      .catch(() => {
        if (press === current) {
          cancelPress(current);
        }
      });
  };

  /**
   * Opens a press on a resolved target: checks the domain, composites the sample and starts the worker. Null when
   * the press does nothing; the caller then cancels the target.
   */
  const startPress = (
    ctx: ToolContext,
    input: PointerInput,
    target: PaintTarget,
    options: PaintBucketOptions
  ): Press | null => {
    const floodFill = ctx.floodFill;
    const erase = options.mode === 'erase';
    if (erase && target.transparencyLocked) {
      // Erasing would change the locked alpha, as the Eraser refuses.
      const layer = getDocumentLayer(ctx.getDocument(), target.layerId);
      return layer ? refusePaintTarget(ctx, layer, 'layer-alpha-locked') : null;
    }
    const matrix = paintTargetMatrix(target);
    const toLocal = matrix ? invert(matrix) : identity();
    // Re-read: resolving may have created the layer or materialized an image in place.
    const doc = ctx.getDocument();
    const leaf = doc ? lookupDocumentLeaf(doc, target.layerId) : null;
    if (!floodFill || !toLocal || !doc || !leaf) {
      return null;
    }
    const mask = isMaskLayer(leaf.layer);
    // A created layer is empty, so the press samples what the user sees.
    const plan = planSample(ctx, doc, options.sample === 'layer' && !target.createdLayer ? leaf.id : null);
    const clipMask = ctx.getSelectionMask?.() ?? null;
    const clipRect = ctx.getStrokeClipRect?.() ?? null;
    const domain = resolveFillDomain({
      clipToFrame: clipRect !== null,
      contentBounds: plan.contentBounds,
      frame: doc.bbox,
      selectionBounds: clipMask?.rect ?? null,
    });
    const seed = domain
      ? { x: Math.floor(input.documentPoint.x) - domain.x, y: Math.floor(input.documentPoint.y) - domain.y }
      : null;
    if (!domain || !seed || seed.x < 0 || seed.y < 0 || seed.x >= domain.width || seed.y >= domain.height) {
      // Outside the selection or the clipped frame: nothing to fill.
      return null;
    }
    if (exceedsFillPixelCap(domain)) {
      ctx.reportEditRefusal?.({ status: 'fill-too-large' });
      return null;
    }
    const reservation = floodFill.reserveSample(domain.width * domain.height * SAMPLE_PEAK_BYTES_PER_PIXEL);
    if (!reservation) {
      ctx.reportEditRefusal?.('over-budget');
      return null;
    }
    let handedOver = false;
    try {
      const sample = compositeSample(ctx, doc, domain, plan);
      const operation: CoverageOperation = erase
        ? 'destination-out'
        : target.transparencyLocked
          ? 'source-atop'
          : 'source-over';
      const coverage = createCoverageEdit(
        ctx,
        { clipMask, clipRect, layerId: target.layerId, operation, toLocal },
        target.edit,
        0
      );
      const params = {
        antiAlias: !mask && options.antiAlias,
        closeGaps: options.closeGaps,
        expand: mask ? options.expandMask : options.expandRaster,
      };
      const current: Press = {
        alpha: mask ? 1 : options.opacity / 100,
        clipMask,
        clipRect,
        color: erase ? '#000000' : mask ? MASK_PAINT_COLOR : ctx.stores.brushOptions.get().color,
        coverage,
        ctx,
        domain,
        drawn: null,
        drawnSequence: -1,
        erase,
        hard: mask,
        id: nextPressId++,
        image: null,
        outstanding: 0,
        params,
        pressTolerance: options.tolerance,
        released: false,
        scrubbing: false,
        startScreen: { ...input.screenPoint },
        surface: null,
        target,
        tolerance: options.tolerance,
      };
      // The sample's buffer moves to the worker here; the service releases the reservation once it has.
      const first = floodFill.begin(
        current.id,
        sample,
        seed,
        { ...params, contiguous: options.contiguous, tolerance: options.tolerance },
        reservation
      );
      handedOver = true;
      track(current, first);
      return current;
    } finally {
      if (!handedOver) {
        reservation.release();
      }
    }
  };

  /** Whether a press would be refused before it starts, without side effects: drives the not-allowed cursor. */
  const wouldRefuse = (ctx: ToolContext): boolean => {
    const leaf = selectedLeaf(ctx.getDocument());
    if (ctx.stores.paintBucketOptions.get().mode === 'fill') {
      return predictPaintRefusal(leaf, 'brush') !== null;
    }
    // Erase has nothing to erase on a selection that would get a new layer.
    return !leaf || !paintsInPlace(leaf) || predictPaintRefusal(leaf, 'eraser') !== null;
  };

  return {
    cancelSettling: () => {
      if (press) {
        cancelPress(press);
      }
    },
    cursor: (ctx) => (!press && wouldRefuse(ctx) ? 'not-allowed' : 'crosshair'),
    id: 'paintBucket',
    onDeactivate: (_ctx, opts) => {
      // A modifier-hold switch keeps a held press; a real switch or disposal ends even a settling one.
      if (press && !opts?.temporary) {
        cancelPress(press);
      }
    },
    onPointerCancel: () => {
      if (press && !press.released) {
        cancelPress(press);
      }
    },
    onPointerDown: (ctx, input) => {
      if (press || (input.buttons & PRIMARY_BUTTON) === 0 || !ctx.floodFill) {
        return;
      }
      const options = ctx.stores.paintBucketOptions.get();
      if (options.mode === 'erase') {
        const leaf = selectedLeaf(ctx.getDocument());
        if (!leaf || !paintsInPlace(leaf)) {
          // Nothing paintable to erase; the cursor already says so.
          return;
        }
      }
      const target = resolvePaintTarget(ctx, options.mode === 'erase' ? 'eraser' : 'brush');
      if (!target) {
        return;
      }
      let started: Press | null = null;
      try {
        started = startPress(ctx, input, target, options);
      } finally {
        if (!started) {
          target.cancel();
        }
      }
      press = started;
      ctx.updateCursor();
    },
    onPointerMove: (ctx, input) => {
      const current = press;
      if (!current || current.released) {
        return;
      }
      const dx = input.screenPoint.x - current.startScreen.x;
      if (!current.scrubbing) {
        const dy = input.screenPoint.y - current.startScreen.y;
        if (Math.hypot(dx, dy) < PAINT_BUCKET_DRAG_THRESHOLD_PX) {
          return;
        }
        current.scrubbing = true;
      }
      const tolerance = Math.min(
        MAX_FILL_TOLERANCE,
        Math.max(0, Math.round(current.pressTolerance + dx / PAINT_BUCKET_PX_PER_TOLERANCE))
      );
      ctx.stores.paintBucketScrub.set({ screenPoint: { ...input.screenPoint }, tolerance });
      ctx.invalidate({ overlay: true });
      if (tolerance !== current.tolerance && ctx.floodFill) {
        current.tolerance = tolerance;
        // The service replaces this press's queued request, so a fast drag costs one region per worker turn.
        track(current, ctx.floodFill.region(current.id, { ...current.params, tolerance }));
      }
    },
    onPointerUp: (ctx) => {
      const current = press;
      if (!current || current.released) {
        return;
      }
      current.released = true;
      // The landed tolerance becomes the option now, so the options form does not snap back while the fill settles.
      if (current.tolerance !== current.pressTolerance) {
        ctx.stores.paintBucketOptions.set({ ...ctx.stores.paintBucketOptions.get(), tolerance: current.tolerance });
      }
      ctx.stores.paintBucketScrub.set(null);
      ctx.invalidate({ overlay: true });
      if (current.outstanding === 0) {
        finishPress(current);
      }
    },
  };
};
