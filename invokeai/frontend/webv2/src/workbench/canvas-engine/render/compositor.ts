/** Draws cached content bottom first, the staged preview between raster and overlay stacks; callers rasterize. */

import type {
  CanvasBlendMode,
  CanvasDocumentContractV3,
  CanvasLayerContract,
  CanvasMaskFillContract,
} from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { OverlayStackKind } from '@workbench/canvas-engine/document/layerStacks';
import type { FrameDamage, Mat2d, Rect } from '@workbench/canvas-engine/types';

import { compileDocumentLeaves, lookupDocumentLeaf } from '@workbench/canvas-engine/document-model/documentModel';
import {
  ALL_OVERLAY_STACKS_SHOWN,
  planScreenComposition,
} from '@workbench/canvas-engine/document-model/screenComposition';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';
import { intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';

import type { DerivedSurfaceCache } from './derivedSurfaceCache';
import type { GroupCompositeScope, GroupSurfaceContent } from './groupCompositeScopes';
import type { LayerCacheEntry, LayerCacheStore } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { renderControlTransparency } from './controlTransparency';
import { collectCompositedGroups, planGroupCompositeScopes } from './groupCompositeScopes';
import { colorizeMask } from './maskFill';

export const CHECKERBOARD_SQUARE_PX = 8;

export const SMOOTHING_MAX_ZOOM = 1;

/** Nearest-neighbor magnification keeps pixels crisp without per-frame bilinear upscaling. */
export const shouldSmoothAtZoom = (zoom: number): boolean => zoom < SMOOTHING_MAX_ZOOM;

export interface CheckerColors {
  a: string;
  b: string;
}

/** Fallback before React supplies theme colors, and for DOM-free callers. */
export const DEFAULT_CHECKER_COLORS: CheckerColors = { a: '#2a2a2a', b: '#363636' };

export const REGION_OVERLAY_ALPHA = 0.5;

/** Relative to the layer's opacity. */
export const CONTROL_OVERLAY_MAP_ALPHA = 0.5;

export interface CompositeOptions {
  backend: RasterBackend;
  /** Clips document layers only, not the background or staged preview. */
  clipRect?: Rect | null;
  /** Document space, above the raster stack and below control layers, masks and regions. */
  stagedPreview?: { surface: RasterSurface; rect: Rect; opacity?: number } | null;
  /** Null disables it, revealing the widget background. */
  checkerboardTile?: RasterSurface | null;
  /** Defaults true for non-viewport callers. */
  imageSmoothing?: boolean;
  /** Clips and culls to the screen union of its regions; absent repaints the whole target. */
  damage?: FrameDamage;
  /** Missing scale or rotation keep committed values, for position-only move previews. */
  transformOverrides?: ReadonlyMap<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  > | null;
  /** Skips an edited text layer drawn by its live portal. */
  skipLayerId?: string | null;
  /** Null means direct solid fill; without a provider only solid fills render. */
  maskPatternTile?: ((style: string, color: string) => RasterSurface | null) | null;
  /** The viewport supplies its backing scale and owns the scratch used to stencil screen-space patterns. */
  screenMask?: {
    dpr: number;
    getScratch: (width: number, height: number) => RasterSurface;
  };
  /** Display-only: pixel-consuming composites must omit them, or the tint becomes generated content. */
  regionOverlays?: boolean;
  /** Layer-local output bounds, drawn through the layer transform. */
  layerPreviews?: ReadonlyMap<string, { surface: RasterSurface; rect: Rect }> | null;
  /** Also suppresses staged and filter previews. */
  isolationLayerId?: string | null;
  /** Overlay stacks the screen draws; absence draws all. An isolated target draws whatever its stack. */
  showOverlayStacks?: Readonly<Record<Exclude<OverlayStackKind, 'vector'>, boolean> & { vector?: boolean }>;
  /**
   * Memoized adjusted raster surface, or null for raw pixels. Optional provider; without it adjustments are
   * ignored. Cache versions prevent per-frame recomputation.
   */
  adjustedSurface?: ((layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null) | null;
  derivedSurfaces?: DerivedSurfaceCache | null;
  /** Layer-local, composed with the layer transform. */
  floatingSelection?: { layerId: string; surface: RasterSurface; rect: Rect; matrix: Mat2d } | null;
  /** Suppressed under isolation. */
  controlView?: {
    layerId: string;
    view: 'original' | 'overlay';
    surface: RasterSurface;
    rect: Rect;
    matrix: Mat2d;
  } | null;
  diagnostics?: CanvasDiagnostics | null;
  /** Ignored unless prepared for this document, isolation, overlay stacks, grouping and overrides. */
  preparation?: CompositePreparation | null;
  /** `null` without drawable content; absent draws members flat. */
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

export const blendToComposite = (mode: CanvasBlendMode): GlobalCompositeOperation =>
  mode === 'normal' ? 'source-over' : (mode as GlobalCompositeOperation);

const isIsolated = (opts: CompositeOptions): boolean =>
  opts.isolationLayerId !== null && opts.isolationLayerId !== undefined;

type TransformOverrides = NonNullable<CompositeOptions['transformOverrides']>;

/** Engines mutate one map in place, so identity cannot tell revisions apart. */
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

const overlayStacksKey = (shown: CompositeOptions['showOverlayStacks']): string => {
  const stacks = shown ?? ALL_OVERLAY_STACKS_SHOWN;
  return `${+stacks.control}${+stacks.inpaint_mask}${+stacks.regional_guidance}${+(stacks.vector !== false)}`;
};

/** Stable while document, isolation, overlay stacks, grouping and overrides are, so callers share it. */
export interface CompositePreparation {
  readonly document: CanvasDocumentContractV3;
  readonly isolationLayerId: string | null;
  readonly overlayStacksKey: string;
  readonly grouped: boolean;
  readonly overridesKey: string;
  readonly leaves: readonly SemanticLeaf[];
  readonly matrices: readonly Mat2d[];
  readonly scopes: readonly GroupCompositeScope[];
  /** A superset is safe; null draws nothing. */
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

export const prepareComposite = (
  doc: CanvasDocumentContractV3,
  opts: Pick<CompositeOptions, 'groupSurface' | 'isolationLayerId' | 'showOverlayStacks' | 'transformOverrides'> = {}
): CompositePreparation => {
  const isolationLayerId = opts.isolationLayerId ?? null;
  const { leaves } = planScreenComposition(compileDocumentLeaves(doc), {
    isolationLayerId,
    showOverlayStacks: opts.showOverlayStacks ?? ALL_OVERLAY_STACKS_SHOWN,
  });
  // Isolation inspects raw members, so scopes are bypassed.
  const grouped = !!opts.groupSurface && isolationLayerId === null;
  const scopes = grouped ? planGroupCompositeScopes(leaves, collectCompositedGroups(doc)) : [];
  const overrides = opts.transformOverrides ?? null;
  return {
    document: doc,
    grouped,
    isolationLayerId,
    leaves,
    matrices: leaves.map((leaf) => effectiveMatrix(leaf, overrides)),
    overlayStacksKey: overlayStacksKey(opts.showOverlayStacks),
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
  preparation.overlayStacksKey === overlayStacksKey(opts.showOverlayStacks) &&
  preparation.grouped === (!!opts.groupSurface && !isIsolated(opts)) &&
  preparation.overridesKey === transformOverridesKey(opts.transformOverrides);

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
  ctx.fillStyle = colors.a;
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = colors.b;
  ctx.fillRect(0, 0, squarePx, squarePx);
  ctx.fillRect(squarePx, squarePx, squarePx, squarePx);
  return tile;
};

/** Identity-transform placement fixes checker cell size and origin during pan and zoom. */
const drawBackground = (ctx: Ctx, tile: RasterSurface | null, bounds: Rect): void => {
  if (!tile) {
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

/** Rounded outward with a one-pixel pad for antialiased seams; unknown layers repaint everything. */
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
  // A degenerate matrix yields NaN, which compares false against every bound; widen rather than skip.
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

const maskPhaseHash = (layerId: string): number => {
  let hash = 2166136261;
  for (let index = 0; index < layerId.length; index += 1) {
    hash = Math.imul(hash ^ layerId.charCodeAt(index), 16777619);
  }
  return hash;
};

type DrawMaskFill = (
  layerId: string,
  fill: CanvasMaskFillContract,
  coverage: { surface: RasterSurface; rect: Rect; version: number },
  matrix: Mat2d,
  kind: 'mask-fill' | 'region-fill'
) => void;

const drawCachedLayer = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  matrix: Mat2d,
  entry: LayerCacheEntry,
  view: Mat2d,
  opts: CompositeOptions,
  drawMaskFill: DrawMaskFill,
  alpha = 1
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity * alpha;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
  const screenMatrix = multiply(view, matrix);
  setTransformFromMat(ctx, screenMatrix);

  const origin = { x: entry.rect.x, y: entry.rect.y };
  const preview = isIsolated(opts) ? null : (opts.layerPreviews?.get(layer.id) ?? null);
  if (preview) {
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
    drawMaskFill(layer.id, layer.mask.fill, entry, screenMatrix, 'mask-fill');
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
    const adjusted = layer.type === 'raster' && opts.adjustedSurface ? opts.adjustedSurface(layer, entry) : null;
    ctx.drawImage((adjusted ?? entry.surface).canvas, origin.x, origin.y);
  }

  if (opts.regionOverlays && layer.type === 'raster' && layer.inpaint?.isEnabled && !preview && !isIsolated(opts)) {
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = REGION_OVERLAY_ALPHA;
    drawMaskFill(layer.id, layer.inpaint.fill, entry, screenMatrix, 'region-fill');
  }

  ctx.restore();
};

/** Drawn through the layer matrix then their own, never resampled into document space and back. */
const drawLayerLocalSurface = (
  ctx: Ctx,
  leaf: SemanticLeaf,
  matrix: Mat2d,
  view: Mat2d,
  pixels: { surface: RasterSurface; rect: Rect; matrix: Mat2d }
): void => {
  const { layer } = leaf;
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);
  setTransformFromMat(ctx, multiply(multiply(view, matrix), pixels.matrix));
  ctx.drawImage(pixels.surface.canvas, pixels.rect.x, pixels.rect.y);
  ctx.restore();
};

const unionLanding = (
  bounds: Rect | null,
  matrix: Mat2d,
  pixels: { rect: Rect; matrix: Mat2d } | null | undefined
): Rect | null => {
  if (!pixels || isEmpty(pixels.rect)) {
    return bounds;
  }
  const landing = transformBounds(multiply(matrix, pixels.matrix), pixels.rect);
  return bounds ? union(bounds, landing) : landing;
};

const controlViewOf = (leaf: SemanticLeaf, opts: CompositeOptions): CompositeOptions['controlView'] =>
  !isIsolated(opts) && leaf.layer.type === 'control' && opts.controlView?.layerId === leaf.id ? opts.controlView : null;

const leafBounds = (leaf: SemanticLeaf, matrix: Mat2d, committed: Rect | null, opts: CompositeOptions): Rect | null => {
  const preview = isIsolated(opts) ? null : (opts.layerPreviews?.get(leaf.id) ?? null);
  let bounds = committed;
  if (preview) {
    const previewBounds = transformBounds(matrix, preview.rect);
    bounds = bounds ? union(bounds, previewBounds) : previewBounds;
  }
  const float = !isIsolated(opts) && opts.floatingSelection?.layerId === leaf.id ? opts.floatingSelection : null;
  return unionLanding(unionLanding(bounds, matrix, float), matrix, controlViewOf(leaf, opts));
};

/** Non-finite geometry counts as visible. */
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
  const drawMaskFill: DrawMaskFill = (layerId, fill, coverage, matrix, kind) => {
    const { surface, rect, version } = coverage;
    const tile = opts.maskPatternTile?.(fill.style, fill.color) ?? null;
    if (tile && opts.screenMask) {
      const { dpr, getScratch } = opts.screenMask;
      const area = intersect(roundOut(transformBounds(matrix, rect)), repaint);
      if (!area || isEmpty(area) || ![area.x, area.y, area.width, area.height].every(Number.isFinite)) {
        return;
      }
      const scratch = getScratch(area.width, area.height);
      const stencil = scratch.ctx;
      stencil.save();
      identityTransform(stencil);
      stencil.clearRect(0, 0, area.width, area.height);
      // The retained scratch can be far larger than a small repaint; drawing the whole mask across it is the cost.
      stencil.beginPath();
      stencil.rect(0, 0, area.width, area.height);
      stencil.clip();
      stencil.globalCompositeOperation = 'source-over';
      stencil.imageSmoothingEnabled = ctx.imageSmoothingEnabled;
      setTransformFromMat(stencil, { ...matrix, e: matrix.e - area.x, f: matrix.f - area.y });
      stencil.drawImage(surface.canvas, rect.x, rect.y);

      // Give each layer a stable phase in viewport CSS coordinates, independent of damage and transforms.
      const phase = maskPhaseHash(layerId);
      const offsetX = ((phase & 0xffff) / 0x10000) * tile.width;
      const offsetY = ((phase >>> 16) / 0x10000) * tile.height;
      stencil.imageSmoothingEnabled = true;
      stencil.setTransform(dpr, 0, 0, dpr, offsetX * dpr - area.x, offsetY * dpr - area.y);
      stencil.globalCompositeOperation = 'source-in';
      stencil.fillStyle = stencil.createPattern(tile.canvas, 'repeat') ?? fill.color;
      stencil.fillRect(area.x / dpr - offsetX, area.y / dpr - offsetY, area.width / dpr, area.height / dpr);
      stencil.restore();

      ctx.save();
      identityTransform(ctx);
      ctx.drawImage(scratch.canvas, 0, 0, area.width, area.height, area.x, area.y, area.width, area.height);
      ctx.restore();
      return;
    }

    const colorized = opts.derivedSurfaces
      ? opts.derivedSurfaces.get({
          create: (target) => colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile, target),
          kind,
          layerId,
          paramsKey: `${fill.style}:${fill.color}`,
          source: surface,
          sourceVersion: version,
        })
      : colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile);
    ctx.drawImage(colorized.canvas, rect.x, rect.y);
  };

  ctx.save();
  // Nested layer restores preserve it.
  ctx.imageSmoothingEnabled = opts.imageSmoothing ?? true;

  // Unchanged pixels outside the repaint region keep the previous frame.
  identityTransform(ctx);
  if (damage.kind === 'rect') {
    ctx.beginPath();
    ctx.rect(repaint.x, repaint.y, repaint.width, repaint.height);
    ctx.clip();
  }
  ctx.clearRect(repaint.x, repaint.y, repaint.width, repaint.height);
  drawBackground(ctx, opts.checkerboardTile ?? null, repaint);

  const applyClipRect = (): void => {
    if (opts.clipRect) {
      ctx.save();
      setTransformFromMat(ctx, view);
      ctx.beginPath();
      ctx.rect(opts.clipRect.x, opts.clipRect.y, opts.clipRect.width, opts.clipRect.height);
      ctx.clip();
    }
  };
  applyClipRect();

  // Lands once, unclipped, before the first overlay-stack leaf (or after every leaf).
  const staged = isIsolated(opts) ? null : (opts.stagedPreview ?? null);
  let stagedPending = staged !== null;
  const drawStaged = (): void => {
    stagedPending = false;
    if (!staged || !reaches(staged.rect, view, repaint)) {
      return;
    }
    if (opts.clipRect) {
      ctx.restore();
    }
    ctx.save();
    setTransformFromMat(ctx, view);
    ctx.globalAlpha = staged.opacity ?? 1;
    ctx.drawImage(staged.surface.canvas, staged.rect.x, staged.rect.y, staged.rect.width, staged.rect.height);
    ctx.restore();
    applyClipRect();
  };

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
    const controlView = controlViewOf(leaf, opts);
    if (controlView) {
      drawLayerLocalSurface(ctx, leaf, matrix, view, controlView);
    }
    // A float still draws over an emptied source: its pixels are detached.
    if (entry && entry.rect.width > 0 && entry.rect.height > 0 && controlView?.view !== 'original') {
      drawCachedLayer(ctx, leaf, matrix, entry, view, opts, drawMaskFill, controlView ? CONTROL_OVERLAY_MAP_ALPHA : 1);
      opts.diagnostics?.increment('layersDrawn');
    }
    if (float?.layerId === leaf.id) {
      drawLayerLocalSurface(ctx, leaf, matrix, view, float);
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
    if (stagedPending && leaves[index]!.stack !== 'raster') {
      drawStaged();
    }
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
      // Display-only, so drawn above the group composite rather than baked into its memo.
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
          const matrix = multiply(view, matrices[member]!);
          setTransformFromMat(ctx, matrix);
          ctx.globalCompositeOperation = 'source-over';
          ctx.globalAlpha = REGION_OVERLAY_ALPHA;
          drawMaskFill(layer.id, layer.inpaint.fill, memberEntry, matrix, 'region-fill');
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

  if (stagedPending) {
    drawStaged();
  }
  if (opts.clipRect) {
    ctx.restore();
  }

  ctx.restore();
};
