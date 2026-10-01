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
import type { FrameDamage, Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves, lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import {
  ALL_OVERLAY_STACKS_SHOWN,
  planScreenComposition,
} from '@workbench/canvas-engine/document-model/screenComposition';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';

import type { DerivedSurfaceCache } from './derivedSurfaceCache';
import type { GroupCompositeScope, GroupSurfaceContent } from './groupCompositeScopes';
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

/** Opacity of the display-only regenerate tint drawn over a raster layer's coverage. */
export const REGION_OVERLAY_ALPHA = 0.5;

/** Dashed outline drawn around a staged-generation preview so it reads as pending, not committed. */
const STAGED_PREVIEW_OUTLINE_COLOR = '#3b82f6';
const STAGED_PREVIEW_OUTLINE_WIDTH = 2;
const STAGED_PREVIEW_OUTLINE_DASH = 6;

/** Optional inputs to {@link compositeDocument}. */
export interface CompositeOptions {
  /** Allocates mask, control-effect and group intermediates; every composite renders through it. */
  backend: RasterBackend;
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
   * Bounded damage clips clear, checkerboard and every draw to the screen union of its regions and culls
   * contributors outside it; damage resolving offscreen draws nothing. Absence repaints the whole target.
   */
  damage?: FrameDamage;
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
  /**
   * Reused frame description from {@link prepareComposite}; ignored unless it was prepared for this document,
   * isolation, grouping and transform overrides.
   */
  preparation?: CompositePreparation | null;
  /**
   * An isolated group's document-space composite (stack applied), `null` when it has no drawable content. Filter
   * previews and the float of its members draw inside it. Absent ⇒ members draw flat.
   */
  groupSurface?:
    | ((
        scope: GroupCompositeScope,
        members: readonly SemanticLeaf[],
        memberMatrices: readonly Mat2d[],
        content: GroupSurfaceContent
      ) => { surface: RasterSurface; rect: Rect } | null)
    | null;
}

type Ctx = RasterSurface['ctx'];

/** Maps a document blend mode to the canvas `globalCompositeOperation`. Also used by `colorSample.ts`. */
export const blendToComposite = (mode: CanvasBlendMode): GlobalCompositeOperation =>
  mode === 'normal' ? 'source-over' : (mode as GlobalCompositeOperation);

const isIsolated = (opts: CompositeOptions): boolean =>
  opts.isolationLayerId !== null && opts.isolationLayerId !== undefined;

type TransformOverrides = NonNullable<CompositeOptions['transformOverrides']>;

/** A stable key for override contents; engines mutate one map in place, so identity cannot tell revisions apart. */
export const transformOverridesKey = (overrides: TransformOverrides | null | undefined): string => {
  if (!overrides || overrides.size === 0) {
    return '';
  }
  let key = '';
  for (const [id, o] of overrides) {
    key += `${id}:${o.x},${o.y},${o.scaleX ?? '-'},${o.scaleY ?? '-'},${o.rotation ?? '-'};`;
  }
  return key;
};

/**
 * One frame description: the leaves drawn (bottom first), their effective matrices with overrides applied, the
 * isolated group scopes, and optionally each leaf's document-space bounds this frame. Stable while the document,
 * isolation, grouping and override contents are, so frame demand, compositing and sampling share it.
 */
export interface CompositePreparation {
  readonly document: CanvasDocumentContractV3;
  readonly isolationLayerId: string | null;
  readonly grouped: boolean;
  readonly overridesKey: string;
  readonly leaves: readonly SemanticLeaf[];
  readonly matrices: readonly Mat2d[];
  readonly scopes: readonly GroupCompositeScope[];
  /** Document-space bounds a leaf's pixels can cover this frame (a superset is safe); null bounds draw nothing. */
  readonly bounds?: readonly (Rect | null)[];
}

const effectiveMatrix = (leaf: SemanticLeaf, overrides: TransformOverrides | null | undefined): Mat2d => {
  const override = overrides?.get(leaf.id);
  if (!override) {
    return leaf.worldTransform;
  }
  const { layer } = leaf;
  return fromTRS(
    { x: override.x, y: override.y },
    override.rotation ?? layer.transform.rotation,
    override.scaleX ?? layer.transform.scaleX,
    override.scaleY ?? layer.transform.scaleY
  );
};

/** Plans the draw order, placement and isolated group scopes for `doc`, for callers compositing it repeatedly. */
export const prepareComposite = (
  doc: CanvasDocumentContractV3,
  opts: Pick<CompositeOptions, 'groupSurface' | 'isolationLayerId' | 'transformOverrides'> = {}
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
  const overrides = opts.transformOverrides ?? null;
  return {
    document: doc,
    grouped,
    isolationLayerId,
    leaves,
    matrices: leaves.map((leaf) => effectiveMatrix(leaf, overrides)),
    overridesKey: transformOverridesKey(overrides),
    scopes,
  };
};

const isPreparedFor = (
  preparation: CompositePreparation | null | undefined,
  doc: CanvasDocumentContractV3,
  opts: CompositeOptions
): preparation is CompositePreparation =>
  !!preparation &&
  preparation.document === doc &&
  preparation.isolationLayerId === (opts.isolationLayerId ?? null) &&
  preparation.grouped === (!!opts.groupSurface && !isIsolated(opts)) &&
  preparation.overridesKey === transformOverridesKey(opts.transformOverrides);

/** Reuses `previous` while it still describes `doc` under these options, else prepares a new description. */
export const reusePreparation = (
  previous: CompositePreparation | null | undefined,
  doc: CanvasDocumentContractV3,
  opts: CompositeOptions
): CompositePreparation => (isPreparedFor(previous, doc, opts) ? previous : prepareComposite(doc, opts));

const setTransformFromMat = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const identityTransform = (ctx: Ctx): void => {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
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

type ScreenDamage = { kind: 'full' } | { kind: 'none' } | { kind: 'rect'; rect: Rect };

/**
 * Transform local damage by view*layerMatrix into screen bounds, rounded outward with a one-pixel pad for
 * antialiased seams. Unknown layers or non-finite geometry repaint everything; regions off the target repaint
 * nothing.
 */
const resolveDamage = (
  plan: CompositePreparation,
  view: Mat2d,
  target: RasterSurface,
  damage: FrameDamage | undefined
): ScreenDamage => {
  if (!damage || damage.kind === 'full') {
    return { kind: 'full' };
  }
  if (damage.kind === 'none') {
    return { kind: 'none' };
  }
  let accumulated: Rect | null = null;
  for (const region of damage.regions) {
    const index = plan.leaves.findIndex((leaf) => leaf.id === region.layerId);
    const leaf = index >= 0 ? plan.leaves[index] : lookupDocumentLeaf(plan.document, region.layerId);
    if (!leaf) {
      return { kind: 'full' };
    }
    const matrix = index >= 0 ? plan.matrices[index]! : leaf.worldTransform;
    const bounds = transformBounds(multiply(view, matrix), region.rect);
    accumulated = accumulated ? union(accumulated, bounds) : bounds;
  }
  if (!accumulated) {
    return { kind: 'none' };
  }
  const grown = roundOut({
    height: accumulated.height + 2,
    width: accumulated.width + 2,
    x: accumulated.x - 1,
    y: accumulated.y - 1,
  });
  // A degenerate matrix yields NaN, which compares false against every bound; widen rather than skip drawing.
  if (![grown.x, grown.y, grown.width, grown.height].every(Number.isFinite)) {
    return { kind: 'full' };
  }
  const visible = intersect(grown, { height: target.height, width: target.width, x: 0, y: 0 });
  return visible && !isEmpty(visible) ? { kind: 'rect', rect: visible } : { kind: 'none' };
};

const isMaskLayer = (
  layer: CanvasLayerContract
): layer is Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }> =>
  layer.type === 'regional_guidance' || layer.type === 'inpaint_mask';

/** Colorize mask alpha with fill/pattern on a source-in intermediate, then draw with the current layer transform. */
const drawMaskLayer = (
  ctx: Ctx,
  layer: Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }>,
  surface: RasterSurface,
  sourceVersion: number,
  origin: Vec2,
  opts: CompositeOptions
): void => {
  const fill = layer.mask.fill;
  const tile = opts.maskPatternTile ? opts.maskPatternTile(fill.style, fill.color) : null;
  const colorized = opts.derivedSurfaces
    ? opts.derivedSurfaces.get({
        create: (target) => colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile, target),
        kind: 'mask-fill',
        layerId: layer.id,
        paramsKey: `${fill.style}:${fill.color}`,
        source: surface,
        sourceVersion,
      })
    : colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile);
  // Draw at local content origin; the outer context already applies layer opacity.
  ctx.drawImage(colorized.canvas, origin.x, origin.y);
};

const drawCachedLayer = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  matrix: Mat2d,
  entry: LayerCacheEntry,
  view: Mat2d,
  opts: CompositeOptions
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
  setTransformFromMat(ctx, multiply(view, matrix));

  // The cache surface holds pixels for `entry.rect` in layer-local space; draw
  // it at that local origin (offset paint/mask layers place their content off-zero).
  const origin = { x: entry.rect.x, y: entry.rect.y };
  const preview = isIsolated(opts) ? null : (opts.layerPreviews?.get(layer.id) ?? null);
  if (preview) {
    // Draw the full filter output at its local rect with the same control-transparency display effect as committed
    // pixels.
    const displayPreview =
      layer.type === 'control' && layer.withTransparencyEffect
        ? opts.derivedSurfaces
          ? opts.derivedSurfaces.get({
              create: (target) =>
                renderControlTransparency(
                  opts.backend,
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
  } else if (layer.type === 'control' && layer.withTransparencyEffect) {
    const effect = opts.derivedSurfaces
      ? opts.derivedSurfaces.get({
          create: (target) =>
            renderControlTransparency(opts.backend, entry.surface, entry.surface.width, entry.surface.height, target),
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
  ctx.globalAlpha = REGION_OVERLAY_ALPHA;
  const tile = opts.maskPatternTile ? opts.maskPatternTile(fill.style, fill.color) : null;
  const colorized = opts.derivedSurfaces
    ? opts.derivedSurfaces.get({
        create: (target) =>
          colorizeMask(
            opts.backend,
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
};

/**
 * Draws the floating selection over the layer it was cut from. Its pixels are
 * layer-local, so they go through the layer's (possibly overridden) matrix and
 * then the float's own — never resampled into document space and back.
 */
const drawFloatingSelection = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  matrix: Mat2d,
  view: Mat2d,
  float: NonNullable<CompositeOptions['floatingSelection']>
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
  setTransformFromMat(ctx, multiply(multiply(view, matrix), float.matrix));
  ctx.drawImage(float.surface.canvas, float.rect.x, float.rect.y);
  ctx.restore();
};

/** Document-space bounds a leaf can cover this frame: committed pixels joined with its preview and float. */
const leafBounds = (leaf: SemanticLeaf, matrix: Mat2d, committed: Rect | null, opts: CompositeOptions): Rect | null => {
  const preview = isIsolated(opts) ? null : (opts.layerPreviews?.get(leaf.id) ?? null);
  let bounds = committed;
  if (preview) {
    const previewBounds = transformBounds(matrix, preview.rect);
    bounds = bounds ? union(bounds, previewBounds) : previewBounds;
  }
  const float = !isIsolated(opts) && opts.floatingSelection?.layerId === leaf.id ? opts.floatingSelection : null;
  if (float && !isEmpty(float.rect)) {
    const landing = transformBounds(multiply(matrix, float.matrix), float.rect);
    bounds = bounds ? union(bounds, landing) : landing;
  }
  return bounds;
};

/** Whether document-space `bounds` reach the screen region; non-finite geometry counts as visible. */
const reaches = (bounds: Rect | null, view: Mat2d, region: Rect): boolean => {
  if (!bounds) {
    return false;
  }
  const screen = transformBounds(view, bounds);
  if (![screen.x, screen.y, screen.width, screen.height].every(Number.isFinite)) {
    return true;
  }
  return intersect(screen, region) !== null;
};

export const compositeDocument = (
  target: RasterSurface,
  doc: CanvasDocumentContractV3,
  caches: LayerCacheStore,
  view: Mat2d,
  opts: CompositeOptions
): void => {
  const plan = reusePreparation(opts.preparation, doc, opts);
  const targetRect: Rect = { height: target.height, width: target.width, x: 0, y: 0 };
  const damage = resolveDamage(plan, view, target, opts.damage);
  if (damage.kind === 'none') {
    return;
  }
  const ctx = target.ctx;
  opts.diagnostics?.increment('compositeFrames');
  const repaint = damage.kind === 'rect' ? damage.rect : targetRect;

  ctx.save();
  // Set smoothing once under the outer save; nested layer restores preserve it.
  ctx.imageSmoothingEnabled = opts.imageSmoothing ?? true;

  // Clear and checker-fill the repaint region, clipping every draw to it. Unchanged pixels keep the previous frame.
  identityTransform(ctx);
  if (damage.kind === 'rect') {
    ctx.beginPath();
    ctx.rect(repaint.x, repaint.y, repaint.width, repaint.height);
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

  const { leaves, matrices, scopes } = plan;
  const boundsAt = (index: number): Rect | null => {
    const leaf = leaves[index]!;
    const matrix = matrices[index]!;
    let committed = plan.bounds?.[index];
    if (committed === undefined) {
      const entry = caches.peek(leaf.id);
      committed = entry && !isEmpty(entry.rect) ? transformBounds(matrix, entry.rect) : null;
    }
    return leafBounds(leaf, matrix, committed, opts);
  };
  const previews = isIsolated(opts) ? null : (opts.layerPreviews ?? null);
  const float = isIsolated(opts) ? null : (opts.floatingSelection ?? null);
  let scopeIndex = 0;

  const drawLeafFlat = (index: number): void => {
    const leaf = leaves[index]!;
    opts.diagnostics?.increment('layersConsidered');
    if (!reaches(boundsAt(index), view, repaint)) {
      opts.diagnostics?.increment('layersCulled');
      return;
    }
    const matrix = matrices[index]!;
    const entry = caches.get(leaf.id);
    // A float still draws over an emptied source: its pixels are detached.
    if (entry && entry.rect.width > 0 && entry.rect.height > 0) {
      drawCachedLayer(ctx, leaf, matrix, entry, view, opts);
      opts.diagnostics?.increment('layersDrawn');
    }
    if (float?.layerId === leaf.id) {
      drawFloatingSelection(ctx, leaf, matrix, view, float);
    }
  };

  const content = (start: number, end: number): GroupSurfaceContent => {
    const excludeIds = new Set<string>();
    for (let index = start; index < end; index += 1) {
      if (leaves[index]!.id === opts.skipLayerId) {
        excludeIds.add(leaves[index]!.id);
      }
    }
    return { excludeIds, float, previews };
  };

  for (let index = 0; index < leaves.length; index += 1) {
    const scope = scopeIndex < scopes.length ? scopes[scopeIndex]! : null;
    if (scope && index === scope.start) {
      scopeIndex += 1;
      let scopeBounds: Rect | null = null;
      for (let member = scope.start; member < scope.end; member += 1) {
        const bounds = leaves[member]!.id === opts.skipLayerId ? null : boundsAt(member);
        scopeBounds = bounds ? (scopeBounds ? union(scopeBounds, bounds) : bounds) : scopeBounds;
      }
      if (!reaches(scopeBounds, view, repaint)) {
        opts.diagnostics?.increment('layersCulled');
        index = scope.end - 1;
        continue;
      }
      const members = leaves.slice(scope.start, scope.end);
      const memberMatrices = matrices.slice(scope.start, scope.end);
      const result = opts.groupSurface!(scope, members, memberMatrices, content(scope.start, scope.end));
      if (result) {
        ctx.save();
        ctx.globalAlpha = scope.opacity;
        ctx.globalCompositeOperation = blendToComposite(scope.blendMode);
        setTransformFromMat(ctx, view);
        ctx.drawImage(result.surface.canvas, result.rect.x, result.rect.y);
        ctx.restore();
        opts.diagnostics?.increment('layersDrawn');
      }
      // Region overlays are display-only, so they ride ABOVE the group
      // composite rather than being baked into (and staled with) its memo.
      for (let member = scope.start; opts.regionOverlays && member < scope.end; member += 1) {
        const { layer } = leaves[member]!;
        if (
          layer.id === opts.skipLayerId ||
          previews?.has(layer.id) ||
          layer.type !== 'raster' ||
          !layer.inpaint?.isEnabled
        ) {
          continue;
        }
        const memberEntry = caches.get(layer.id);
        if (memberEntry && memberEntry.rect.width > 0 && memberEntry.rect.height > 0) {
          ctx.save();
          setTransformFromMat(ctx, multiply(view, matrices[member]!));
          drawRegionCoverage(ctx, layer.id, layer.inpaint.fill, memberEntry, opts);
          ctx.restore();
        }
      }
      index = scope.end - 1;
      continue;
    }
    if (leaves[index]!.id === opts.skipLayerId) {
      continue;
    }
    drawLeafFlat(index);
  }

  if (opts.clipRect) {
    ctx.restore();
  }

  // Draw staged preview in document space with a dashed outline distinguishing pending pixels.
  const staged = isIsolated(opts) ? null : opts.stagedPreview;
  // Half the outline lands outside the rect, so a repaint beside it must still redraw that half.
  if (staged && reaches(staged.rect, view, expand(repaint, STAGED_PREVIEW_OUTLINE_WIDTH))) {
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
