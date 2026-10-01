/**
 * Draws cached content: clear/checkerboard across the viewport, bottom-first transformed layers with
 * opacity/blend, then staged preview. Missing caches are skipped; callers own rasterization. RasterSurface
 * contexts make ordering testable.
 */

import type {
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasMaskFillContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { LayerDamage, Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves, lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import {
  ALL_OVERLAY_STACKS_SHOWN,
  planScreenComposition,
} from '@workbench/canvas-engine/document-model/screenComposition';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';

import type { DerivedSurfaceCache } from './derivedSurfaceCache';
import type { GroupCompositeScope } from './groupCompositeScopes';
import type { LayerCacheEntry, LayerCacheStore } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { renderControlTransparency } from './controlTransparency';
import { collectCompositedGroups, planGroupCompositeScopes } from './groupCompositeScopes';
import { colorizeMask } from './maskFill';

/** Screen-space size (px) of each checkerboard square for transparent backgrounds. */
export const CHECKERBOARD_SQUARE_PX = 8;

export const SMOOTHING_MAX_ZOOM = 1;

/**
 * Smooth only below 1x for clean downscaling. Magnification uses nearest-neighbor for crisp pixels without
 * per-frame bilinear upscaling.
 */
export const shouldSmoothAtZoom = (zoom: number): boolean => zoom < SMOOTHING_MAX_ZOOM;

/** The two square colors of the transparency checkerboard. */
export interface CheckerColors {
  /** The base color, filled across the whole tile. */
  a: string;
  /** The alternating color, drawn on the tile's diagonal cells. */
  b: string;
}

/** Dark checker fallback before React supplies semantic theme colors, and for DOM-free callers. */
export const DEFAULT_CHECKER_COLORS: CheckerColors = { a: '#2a2a2a', b: '#363636' };

/**
 * Without a backend, approximate masks with coverage plus flat tint. Production uses source-in colorization at
 * layer opacity.
 */
export const MASK_TINT_ALPHA = 0.5;

/** Dashed outline drawn around a staged-generation preview so it reads as pending, not committed. */
const STAGED_PREVIEW_OUTLINE_COLOR = '#3b82f6';
const STAGED_PREVIEW_OUTLINE_WIDTH = 2;
const STAGED_PREVIEW_OUTLINE_DASH = 6;

/** Optional inputs to {@link compositeDocument}. */
export interface CompositeOptions {
  /** Clips document layers to this document-space rect without clipping the background or staged preview. */
  clipRect?: Rect | null;
  /** A staged generation candidate to draw at its placement (document space). */
  stagedPreview?: { surface: RasterSurface; rect: Rect; opacity?: number } | null;
  /**
   * Checker tile fills the unbounded viewport. Null/absence disables it, revealing themed widget background
   * through the cleared surface.
   */
  checkerboardTile?: RasterSurface | null;
  /**
   * Smoothing defaults true for non-viewport callers; engine frames use {@link shouldSmoothAtZoom} for crisp
   * magnification.
   */
  imageSmoothing?: boolean;
  /**
   * Layer-local damage clips clear, checkerboard and layer draws to its screen union, retaining other pixels.
   * Null/absence repaints the whole target.
   */
  damage?: LayerDamage[] | null;
  /**
   * Transient transforms override committed values without changing the mirror. Missing scale/rotation retain
   * committed values, supporting position-only move previews.
   */
  transformOverrides?: ReadonlyMap<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  > | null;
  /** Skip an edited text layer to avoid drawing beneath its live portal; null/absence skips nothing. */
  skipLayerId?: string | null;
  /**
   * Backend supplies mask source-in intermediates. Absence uses {@link MASK_TINT_ALPHA} approximation; production
   * always supplies it.
   */
  backend?: RasterBackend | null;
  /** Cached mask pattern by style/color; null means direct solid fill. Without a provider, only solid fills render. */
  maskPatternTile?: ((style: string, color: string) => RasterSurface | null) | null;
  /**
   * Display-only regenerate overlays for screen/Overview. Pixel-consuming composites must omit them or their tint
   * becomes generated, sampled or exported content.
   */
  regionOverlays?: boolean;
  /**
   * Filter previews replace committed pixels at layer-local output bounds through the layer transform without
   * document mutation. Null/absence means none.
   */
  layerPreviews?: ReadonlyMap<string, { surface: RasterSurface; rect: Rect }> | null;
  /** Isolate one leaf and suppress staged/filter previews; disabled or hidden leaves remain undrawn. */
  isolationLayerId?: string | null;
  /**
   * Memoized adjusted raster surface, or null for raw pixels. Optional provider; without it adjustments are
   * ignored. Cache versions prevent per-frame recomputation.
   */
  adjustedSurface?: ((layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null) | null;
  /** Shared memoized surfaces for mask and control display effects. */
  derivedSurfaces?: DerivedSurfaceCache | null;
  /**
   * Draw floating pixels immediately above their cut source layer with its opacity/blend. Surface, rect and float
   * matrix are layer-local and compose with the layer transform.
   */
  floatingSelection?: { layerId: string; surface: RasterSurface; rect: Rect; matrix: Mat2d } | null;
  /** Optional deterministic render counters; omitted in the normal zero-overhead path. */
  diagnostics?: CanvasDiagnostics | null;
  /** Reused draw plan from {@link prepareComposite}; ignored unless it was prepared for this document and isolation. */
  preparation?: CompositePreparation | null;
  /**
   * An adjusted group's document-space composite (stack applied), `null` when
   * it has no drawable content; `excludeIds` members are left out for the
   * caller to draw separately. Absent ⇒ members draw flat.
   */
  groupSurface?:
    | ((
        scope: GroupCompositeScope,
        members: readonly SemanticLeaf[],
        memberMatrices: readonly Mat2d[],
        excludeIds: ReadonlySet<string>
      ) => { surface: RasterSurface; rect: Rect } | null)
    | null;
}

type Ctx = RasterSurface['ctx'];

/** Maps a document blend mode to the canvas `globalCompositeOperation`. Also used by `colorSample.ts`. */
export const blendToComposite = (mode: CanvasBlendMode): GlobalCompositeOperation =>
  mode === 'normal' ? 'source-over' : (mode as GlobalCompositeOperation);

const isIsolated = (opts: CompositeOptions): boolean =>
  opts.isolationLayerId !== null && opts.isolationLayerId !== undefined;

/** The leaves and group scopes one document composites with; stable while the document and isolation are. */
export interface CompositePreparation {
  readonly document: CanvasDocumentContractV3;
  readonly isolationLayerId: string | null;
  readonly grouped: boolean;
  readonly leaves: readonly SemanticLeaf[];
  readonly scopes: readonly GroupCompositeScope[];
}

/** Plans the draw order and isolated group scopes for `doc`, for callers compositing it repeatedly. */
export const prepareComposite = (
  doc: CanvasDocumentContractV3,
  opts: Pick<CompositeOptions, 'groupSurface' | 'isolationLayerId'> = {}
): CompositePreparation => {
  const isolationLayerId = opts.isolationLayerId ?? null;
  // The plan lists the leaves to draw bottom first; the `stagedPreview` lands on top of every stack.
  const { leaves } = planScreenComposition(compileDocumentLeaves(doc), {
    isolationLayerId,
    showOverlayStacks: ALL_OVERLAY_STACKS_SHOWN,
  });
  // Isolation mode inspects raw members, so scopes are bypassed while active.
  const grouped = !!opts.groupSurface && isolationLayerId === null;
  const scopes = grouped ? planGroupCompositeScopes(leaves, collectCompositedGroups(doc)) : [];
  return { document: doc, grouped, isolationLayerId, leaves, scopes };
};

const isPreparedFor = (
  preparation: CompositePreparation | null | undefined,
  doc: CanvasDocumentContractV3,
  opts: CompositeOptions
): preparation is CompositePreparation =>
  !!preparation &&
  preparation.document === doc &&
  preparation.isolationLayerId === (opts.isolationLayerId ?? null) &&
  preparation.grouped === (!!opts.groupSurface && !isIsolated(opts));

const setTransformFromMat = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const identityTransform = (ctx: Ctx): void => {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
};

const getEffectiveLayerMatrix = (leaf: SemanticLeaf, opts: CompositeOptions): Mat2d => {
  const override = opts.transformOverrides?.get(leaf.id);
  if (!override) {
    return leaf.worldTransform;
  }
  const { layer } = leaf;
  return fromTRS(
    { x: override?.x ?? layer.transform.x, y: override?.y ?? layer.transform.y },
    override?.rotation ?? layer.transform.rotation,
    override?.scaleX ?? layer.transform.scaleX,
    override?.scaleY ?? layer.transform.scaleY
  );
};

const isDefinitelyOffscreen = (
  leaf: SemanticLeaf,
  entry: LayerCacheEntry,
  view: Mat2d,
  target: RasterSurface,
  opts: CompositeOptions
): boolean => {
  const bounds = transformBounds(multiply(view, getEffectiveLayerMatrix(leaf, opts)), entry.rect);
  if (![bounds.x, bounds.y, bounds.width, bounds.height].every(Number.isFinite)) {
    return false;
  }
  return intersect(bounds, { height: target.height, width: target.width, x: 0, y: 0 }) === null;
};

/** Build a reusable 2x2 checker tile through {@link RasterBackend}; frames use patterns instead of per-cell fills. */
export const createCheckerboardTile = (
  backend: RasterBackend,
  colors: CheckerColors = DEFAULT_CHECKER_COLORS,
  squarePx: number = CHECKERBOARD_SQUARE_PX
): RasterSurface => {
  const size = squarePx * 2;
  const tile = backend.createSurface(size, size);
  const ctx = tile.ctx;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, size, size);
  // Base color across the whole tile, then the alternating cells on the diagonal.
  ctx.fillStyle = colors.a;
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = colors.b;
  ctx.fillRect(0, 0, squarePx, squarePx);
  ctx.fillRect(squarePx, squarePx, squarePx, squarePx);
  return tile;
};

/**
 * Fill the unbounded viewport, ignoring document background. Identity-transform placement fixes checker cell
 * size/origin during pan and zoom. Without a tile, the cleared target reveals widget background.
 */
const drawBackground = (ctx: Ctx, tile: RasterSurface | null, bounds: Rect): void => {
  if (!tile) {
    // Checkerboard disabled: leave the cleared surface showing `bg.inset`.
    return;
  }
  const pattern = ctx.createPattern(tile.canvas, 'repeat');
  if (!pattern) {
    return;
  }
  ctx.fillStyle = pattern;
  ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
};

/**
 * Transform local damage by view*layerMatrix into screen bounds. Missing layers force full repaint. Round outward
 * and pad one pixel to cover antialiased seams.
 */
const resolveDamage = (
  doc: CanvasDocumentContractV3,
  view: Mat2d,
  target: RasterSurface,
  opts: CompositeOptions
): Rect | null => {
  const damage = opts.damage;
  if (!damage || damage.length === 0) {
    return null;
  }
  let accumulated: Rect | null = null;
  for (const region of damage) {
    const leaf = lookupDocumentLeaf(doc, region.layerId);
    if (!leaf) {
      return null;
    }
    const bounds = transformBounds(multiply(view, getEffectiveLayerMatrix(leaf, opts)), region.rect);
    accumulated = accumulated ? union(accumulated, bounds) : bounds;
  }
  if (!accumulated) {
    return null;
  }
  const grown = roundOut({
    height: accumulated.height + 2,
    width: accumulated.width + 2,
    x: accumulated.x - 1,
    y: accumulated.y - 1,
  });
  // A degenerate matrix would put NaN in here, and a NaN rect compares false
  // against every bound — which would quietly skip the frame's drawing rather
  // than widen it. Repaint everything instead.
  if (
    !Number.isFinite(grown.x) ||
    !Number.isFinite(grown.y) ||
    !Number.isFinite(grown.width) ||
    !Number.isFinite(grown.height)
  ) {
    return null;
  }
  return intersect(grown, { height: target.height, width: target.width, x: 0, y: 0 });
};

const isMaskLayer = (
  layer: CanvasLayerContract
): layer is Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }> =>
  layer.type === 'regional_guidance' || layer.type === 'inpaint_mask';

/**
 * Colorize mask alpha with fill/pattern on a source-in intermediate, then draw with the current layer
 * transform/opacity. Without a backend, approximate with coverage and flat tint.
 */
const drawMaskLayer = (
  ctx: Ctx,
  layer: Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }>,
  surface: RasterSurface,
  sourceVersion: number,
  origin: Vec2,
  opts: CompositeOptions
): void => {
  const fill = layer.mask.fill;
  if (opts.backend) {
    const tile = opts.maskPatternTile ? opts.maskPatternTile(fill.style, fill.color) : null;
    const colorized = opts.derivedSurfaces
      ? opts.derivedSurfaces.get({
          create: (target) => colorizeMask(opts.backend!, surface, surface.width, surface.height, fill, tile, target),
          kind: 'mask-fill',
          layerId: layer.id,
          paramsKey: `${fill.style}:${fill.color}`,
          source: surface,
          sourceVersion,
        })
      : colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile);
    // Draw at local content origin; the outer context already applies layer opacity.
    ctx.drawImage(colorized.canvas, origin.x, origin.y);
    return;
  }
  // Backend-less fallback (bare test call): coverage + flat translucent fill.
  ctx.drawImage(surface.canvas, origin.x, origin.y);
  ctx.globalAlpha = layer.opacity * MASK_TINT_ALPHA;
  ctx.fillStyle = fill.color;
  ctx.fillRect(origin.x, origin.y, surface.width, surface.height);
};

const drawCachedLayer = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  entry: LayerCacheEntry,
  view: Mat2d,
  opts: CompositeOptions
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);

  const layerMat = getEffectiveLayerMatrix(leaf, opts);
  setTransformFromMat(ctx, multiply(view, layerMat));

  // The cache surface holds pixels for `entry.rect` in layer-local space; draw
  // it at that local origin (offset paint/mask layers place their content off-zero).
  const origin = { x: entry.rect.x, y: entry.rect.y };
  const preview = isIsolated(opts) ? null : (opts.layerPreviews?.get(layer.id) ?? null);
  if (preview) {
    // Draw the full filter output at its local rect with the same control-transparency display effect as committed
    // pixels.
    const displayPreview =
      layer.type === 'control' && layer.withTransparencyEffect && opts.backend
        ? opts.derivedSurfaces
          ? opts.derivedSurfaces.get({
              create: (target) =>
                renderControlTransparency(
                  opts.backend!,
                  preview.surface,
                  preview.surface.width,
                  preview.surface.height,
                  target
                ),
              kind: 'control-transparency',
              layerId: layer.id,
              paramsKey: 'preview',
              source: preview.surface,
              sourceVersion: 0,
            })
          : renderControlTransparency(opts.backend, preview.surface, preview.surface.width, preview.surface.height)
        : preview.surface;
    ctx.drawImage(displayPreview.canvas, preview.rect.x, preview.rect.y);
  } else if (isMaskLayer(layer)) {
    drawMaskLayer(ctx, layer, entry.surface, entry.version, origin, opts);
  } else if (layer.type === 'control' && layer.withTransparencyEffect && opts.backend) {
    const effect = opts.derivedSurfaces
      ? opts.derivedSurfaces.get({
          create: (target) =>
            renderControlTransparency(opts.backend!, entry.surface, entry.surface.width, entry.surface.height, target),
          kind: 'control-transparency',
          layerId: layer.id,
          paramsKey: 'committed',
          source: entry.surface,
          sourceVersion: entry.version,
        })
      : renderControlTransparency(opts.backend, entry.surface, entry.surface.width, entry.surface.height);
    ctx.drawImage(effect.canvas, origin.x, origin.y);
  } else {
    // Version-keyed adjusted surfaces refresh stroke damage on each cache write and reuse idle frames. Without
    // adjustments/provider, draw raw cache pixels.
    const adjusted = layer.type === 'raster' && opts.adjustedSurface ? opts.adjustedSurface(layer, entry) : null;
    ctx.drawImage((adjusted ?? entry.surface).canvas, origin.x, origin.y);
  }

  if (opts.regionOverlays && layer.type === 'raster' && layer.inpaint?.isEnabled && !preview && !isIsolated(opts)) {
    drawRegionCoverage(ctx, layer.id, layer.inpaint.fill, entry, opts);
  }

  ctx.restore();
};

/**
 * Colorize the raster layer's own alpha above it as a regenerate overlay; strokes, erasure and transforms update
 * coverage without separate pixel persistence.
 */
const drawRegionCoverage = (
  ctx: Ctx,
  layerId: string,
  fill: CanvasMaskFillContract,
  coverage: { surface: RasterSurface; rect: Rect; version: number },
  opts: CompositeOptions
): void => {
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = MASK_TINT_ALPHA;
  if (opts.backend) {
    const tile = opts.maskPatternTile ? opts.maskPatternTile(fill.style, fill.color) : null;
    const colorized = opts.derivedSurfaces
      ? opts.derivedSurfaces.get({
          create: (target) =>
            colorizeMask(
              opts.backend!,
              coverage.surface,
              coverage.surface.width,
              coverage.surface.height,
              fill,
              tile,
              target
            ),
          kind: 'region-fill',
          layerId,
          paramsKey: `${fill.style}:${fill.color}`,
          source: coverage.surface,
          sourceVersion: coverage.version,
        })
      : colorizeMask(opts.backend, coverage.surface, coverage.surface.width, coverage.surface.height, fill, tile);
    ctx.drawImage(colorized.canvas, coverage.rect.x, coverage.rect.y);
    return;
  }
  ctx.fillStyle = fill.color;
  ctx.drawImage(coverage.surface.canvas, coverage.rect.x, coverage.rect.y);
  ctx.fillRect(coverage.rect.x, coverage.rect.y, coverage.rect.width, coverage.rect.height);
};

/**
 * Draws the floating selection over the layer it was cut from. Its pixels are
 * layer-local, so they go through the layer's (possibly overridden) matrix and
 * then the float's own — never resampled into document space and back.
 */
const drawFloatingSelection = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  view: Mat2d,
  opts: CompositeOptions,
  float: NonNullable<CompositeOptions['floatingSelection']>
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
  setTransformFromMat(ctx, multiply(multiply(view, getEffectiveLayerMatrix(leaf, opts)), float.matrix));
  ctx.drawImage(float.surface.canvas, float.rect.x, float.rect.y);
  ctx.restore();
};

export const compositeDocument = (
  target: RasterSurface,
  doc: CanvasDocumentContractV3,
  caches: LayerCacheStore,
  view: Mat2d,
  opts: CompositeOptions = {}
): void => {
  const ctx = target.ctx;
  opts.diagnostics?.increment('compositeFrames');

  ctx.save();

  // Set smoothing once under the outer save; nested layer restores preserve it.
  ctx.imageSmoothingEnabled = opts.imageSmoothing ?? true;

  // Clear and checker-fill the viewport, clipping all draws to declared damage. Unchanged pixels retain the
  // previous frame; empty visible damage leaves the target untouched.
  identityTransform(ctx);
  const damageScreen = resolveDamage(doc, view, target, opts);
  if (damageScreen && isEmpty(damageScreen)) {
    ctx.restore();
    return;
  }
  const repaint: Rect = damageScreen ?? { height: target.height, width: target.width, x: 0, y: 0 };
  if (damageScreen) {
    ctx.beginPath();
    ctx.rect(damageScreen.x, damageScreen.y, damageScreen.width, damageScreen.height);
    ctx.clip();
  }
  ctx.clearRect(repaint.x, repaint.y, repaint.width, repaint.height);
  drawBackground(ctx, opts.checkerboardTile ?? null, repaint);

  if (opts.clipRect) {
    ctx.save();
    setTransformFromMat(ctx, view);
    ctx.beginPath();
    ctx.rect(opts.clipRect.x, opts.clipRect.y, opts.clipRect.width, opts.clipRect.height);
    ctx.clip();
  }

  const plan = isPreparedFor(opts.preparation, doc, opts) ? opts.preparation : prepareComposite(doc, opts);
  const { scopes } = plan;
  let scopeIndex = 0;

  const drawLeafFlat = (leaf: SemanticLeaf): void => {
    opts.diagnostics?.increment('layersConsidered');
    const float = opts.floatingSelection?.layerId === leaf.id ? opts.floatingSelection : null;
    const entry = caches.get(leaf.id);
    // Skip layers with no cache or an empty content rect (a brand-new / cleared
    // paint / mask layer holds no pixels — nothing to draw). A float still draws:
    // its pixels are detached, so an emptied source layer must not hide them.
    if (!entry || entry.rect.width <= 0 || entry.rect.height <= 0) {
      if (float) {
        drawFloatingSelection(ctx, leaf, view, opts, float);
      }
      return;
    }
    if (isDefinitelyOffscreen(leaf, entry, view, target, opts) && !float) {
      opts.diagnostics?.increment('layersCulled');
      return;
    }
    drawCachedLayer(ctx, leaf, entry, view, opts);
    if (float) {
      drawFloatingSelection(ctx, leaf, view, opts, float);
    }
    opts.diagnostics?.increment('layersDrawn');
  };

  for (let index = 0; index < plan.leaves.length; index += 1) {
    const scope = scopeIndex < scopes.length ? scopes[scopeIndex]! : null;
    if (scope && index === scope.start) {
      const members = plan.leaves.slice(scope.start, scope.end);
      const matrices = members.map((member) => getEffectiveLayerMatrix(member, opts));
      // Skip targets and filter previews draw separately, without the group stack.
      const excluded = new Set<string>();
      for (const member of members) {
        if (member.id === opts.skipLayerId || opts.layerPreviews?.has(member.id)) {
          excluded.add(member.id);
        }
      }
      const result = opts.groupSurface!(scope, members, matrices, excluded);
      if (result) {
        ctx.save();
        ctx.globalAlpha = scope.opacity;
        ctx.globalCompositeOperation = blendToComposite(scope.blendMode);
        setTransformFromMat(ctx, view);
        ctx.drawImage(result.surface.canvas, result.rect.x, result.rect.y);
        ctx.restore();
        opts.diagnostics?.increment('layersDrawn');
        // A float whose member is in the composite draws alone.
        for (const member of members) {
          if (member.id === opts.skipLayerId) {
            continue;
          }
          if (excluded.has(member.id)) {
            drawLeafFlat(member);
          } else if (opts.floatingSelection?.layerId === member.id) {
            drawFloatingSelection(ctx, member, view, opts, opts.floatingSelection);
          }
        }
        // Region overlays are display-only, so they ride ABOVE the group
        // composite rather than being baked into (and staled with) its memo.
        for (let memberIndex = 0; opts.regionOverlays && memberIndex < members.length; memberIndex += 1) {
          const member = members[memberIndex]!;
          const { layer } = member;
          if (
            excluded.has(member.id) ||
            member.id === opts.skipLayerId ||
            layer.type !== 'raster' ||
            !layer.inpaint?.isEnabled
          ) {
            continue;
          }
          const memberEntry = caches.get(member.id);
          if (memberEntry && memberEntry.rect.width > 0 && memberEntry.rect.height > 0) {
            ctx.save();
            setTransformFromMat(ctx, multiply(view, matrices[memberIndex]!));
            drawRegionCoverage(ctx, layer.id, layer.inpaint.fill, memberEntry, opts);
            ctx.restore();
          }
        }
        index = scope.end - 1;
        scopeIndex += 1;
        continue;
      }
      scopeIndex += 1;
    }
    const leaf = plan.leaves[index]!;
    if (leaf.id === opts.skipLayerId) {
      continue;
    }
    drawLeafFlat(leaf);
  }

  if (opts.clipRect) {
    ctx.restore();
  }

  // Draw staged preview in document space with a dashed outline distinguishing pending pixels.
  const staged = isIsolated(opts) ? null : opts.stagedPreview;
  if (staged) {
    ctx.save();
    setTransformFromMat(ctx, view);
    ctx.globalAlpha = staged.opacity ?? 1;
    ctx.drawImage(staged.surface.canvas, staged.rect.x, staged.rect.y, staged.rect.width, staged.rect.height);
    // Keep the outline visually constant regardless of zoom by dividing the
    // document-space stroke by the view scale (√det of the linear part).
    const viewScale = Math.sqrt(Math.abs(view.a * view.d - view.b * view.c)) || 1;
    ctx.globalAlpha = 1;
    ctx.strokeStyle = STAGED_PREVIEW_OUTLINE_COLOR;
    ctx.lineWidth = STAGED_PREVIEW_OUTLINE_WIDTH / viewScale;
    ctx.setLineDash([STAGED_PREVIEW_OUTLINE_DASH / viewScale, STAGED_PREVIEW_OUTLINE_DASH / viewScale]);
    ctx.strokeRect(staged.rect.x, staged.rect.y, staged.rect.width, staged.rect.height);
    ctx.setLineDash([]);
    ctx.restore();
  }

  ctx.restore();
};
