/**
 * Composites a canvas document onto a target surface.
 *
 * Draws (in order): a clear + a viewport-filling checkerboard (the unbounded
 * plane's surround; omitted when the checkerboard is off), then each layer's
 * cached surface from bottom to top applying opacity / blend mode / transform,
 * and finally an optional staged-generation preview. Layers without a cache entry are
 * skipped — rasterization is the caller's job (see `rasterizers/` +
 * `layerCache.ts`); this module only draws what's already cached.
 *
 * Every pixel operation flows through the {@link RasterSurface} `ctx`, which
 * in tests is the recording stub, so composite order is assertable in node.
 * Zero React, zero import-time side effects.
 */

import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';
import type { CanvasBlendMode, CanvasDocumentContractV2, CanvasLayerContract } from '@workbench/types';

import { LAYER_GROUP_COUNT, layerGroupRank } from '@workbench/canvas-engine/document/sources';
import { fromTRS, multiply } from '@workbench/canvas-engine/math/mat2d';

import type { LayerCacheEntry, LayerCacheStore } from './layerCache';
import type { RasterBackend, RasterSurface } from './raster';

import { renderControlTransparency } from './controlTransparency';
import { colorizeMask } from './maskFill';

/** Screen-space size (px) of each checkerboard square for transparent backgrounds. */
export const CHECKERBOARD_SQUARE_PX = 8;

/**
 * CSS-zoom threshold at/above which image smoothing is disabled while
 * compositing. See {@link shouldSmoothAtZoom}.
 */
export const SMOOTHING_MAX_ZOOM = 1;

/**
 * Image-smoothing policy for compositing at a given CSS zoom.
 *
 * Smoothing is enabled only when the document is DOWN-scaled (`zoom < 1`) —
 * bilinear interpolation keeps a shrunk image clean. When zoomed IN
 * (`zoom >= 1`) the document is up-scaled to fill the screen; smoothing is
 * disabled so (a) pixels stay crisp when magnified — the behavior legacy pixel
 * editors adopt above ~1× — and (b) the browser skips the per-frame bilinear
 * interpolation of an ever-larger upscale, whose fill-rate cost is precisely
 * what grows with zoom. Nearest-neighbor upscaling is dramatically cheaper.
 */
export const shouldSmoothAtZoom = (zoom: number): boolean => zoom < SMOOTHING_MAX_ZOOM;

/** The two square colors of the transparency checkerboard. */
export interface CheckerColors {
  /** The base color, filled across the whole tile. */
  a: string;
  /** The alternating color, drawn on the tile's diagonal cells. */
  b: string;
}

/**
 * Fallback checkerboard colors when no theme tokens have been fed to the engine
 * (node tests, first frame before React resolves the tokens). Deliberately DARK,
 * theme-appropriate greys (in the spirit of the legacy dark transparency pattern)
 * so the indicator reads as "empty" against the dark workbench surface. In the
 * app these are replaced by resolved Chakra semantic tokens (see
 * `widgets/canvas/checkerColors.ts`).
 */
export const DEFAULT_CHECKER_COLORS: CheckerColors = { a: '#2a2a2a', b: '#363636' };

/**
 * Fallback tint alpha for a mask-bearing layer drawn WITHOUT a backend (a bare
 * {@link compositeDocument} call in a minimal test): the mask coverage is blitted
 * and a translucent flat fill laid over it. The real path (with a backend)
 * colorizes the mask alpha via `source-in` at the layer opacity, matching legacy.
 */
export const MASK_TINT_ALPHA = 0.5;

/** Dashed outline drawn around a staged-generation preview so it reads as pending, not committed. */
const STAGED_PREVIEW_OUTLINE_COLOR = '#3b82f6';
const STAGED_PREVIEW_OUTLINE_WIDTH = 2;
const STAGED_PREVIEW_OUTLINE_DASH = 6;

/** Optional inputs to {@link compositeDocument}. */
export interface CompositeOptions {
  /** A staged generation candidate to draw at its placement (document space). */
  stagedPreview?: { surface: RasterSurface; rect: Rect; opacity?: number } | null;
  /**
   * The cached checkerboard pattern tile (see {@link createCheckerboardTile}) to
   * fill the ENTIRE viewport with (the canvas is an unbounded plane — the checker
   * is the world, not a document backdrop). Omit or pass `null` to draw NO
   * checkerboard (the "checkerboard off" state) — the cleared surface then shows
   * the widget's themed `bg.inset` through it.
   */
  checkerboardTile?: RasterSurface | null;
  /**
   * Whether `imageSmoothingEnabled` is on for this composite's `drawImage`
   * blits (layer caches + staged preview). Defaults to `true` (the browser
   * default) so non-viewport callers are unaffected; the engine feeds
   * {@link shouldSmoothAtZoom} so zoomed-in frames composite crisp and cheap.
   */
  imageSmoothing?: boolean;
  /**
   * Transient per-layer transform overrides (a live move/transform preview): a
   * layer with an entry here is drawn through the overridden transform instead of
   * its committed one. `scaleX`/`scaleY`/`rotation` fall back to the committed
   * transform when absent (the move tool overrides only `x`/`y`). The mirror stays
   * untouched.
   */
  transformOverrides?: ReadonlyMap<
    string,
    { x: number; y: number; scaleX?: number; scaleY?: number; rotation?: number }
  > | null;
  /**
   * A layer id to SKIP entirely (draw nothing for it). Used while a text-edit
   * session is open: the contenteditable portal shows the layer's live text
   * instead, so drawing its committed pixels underneath would double up. `null`
   * (or absent) skips nothing.
   */
  skipLayerId?: string | null;
  /**
   * The raster backend, needed to colorize mask layers (an intermediate surface
   * holds the alpha stencil while the fill is composited `source-in`). When
   * absent, mask layers fall back to the flat-tint approximation
   * ({@link MASK_TINT_ALPHA}); the engine always supplies it.
   */
  backend?: RasterBackend | null;
  /**
   * Returns a cached repeat tile for a mask fill (style, colour), or `null` for a
   * solid fill (drawn directly). The engine caches tiles by `style:color` (like
   * the checkerboard). Absent ⇒ solid fills only.
   */
  maskPatternTile?: ((style: string, color: string) => RasterSurface | null) | null;
  /**
   * Transient per-layer content previews (a non-destructive control-filter
   * preview): a layer with an entry here draws the preview surface at its returned
   * layer-local output rect, through the layer transform, INSTEAD of
   * its committed cache, so the document is untouched until the filter is applied.
   * `null`/absent ⇒ no previews.
   */
  layerPreviews?: ReadonlyMap<string, { surface: RasterSurface; rect: Rect }> | null;
  /** When set, transiently composites only this layer (even if disabled) and suppresses staged content. */
  onlyLayerId?: string | null;
  /**
   * Returns a raster layer's ADJUSTED cache surface (brightness/contrast/
   * saturation/curves applied), or `null` when the layer has identity (or no)
   * adjustments — in which case the committed cache surface is drawn directly.
   * The engine wires this to a memoizing {@link
   * import('./adjustedSurfaceCache').AdjustedSurfaceCache} so the adjusted pixels
   * are NOT recomputed each frame (see that module). Only consulted for raster
   * layers; absent ⇒ adjustments are ignored (a bare test call draws raw pixels).
   */
  adjustedSurface?: ((layer: CanvasLayerContract, entry: LayerCacheEntry) => RasterSurface | null) | null;
}

type Ctx = RasterSurface['ctx'];

/** Maps a document blend mode to the canvas `globalCompositeOperation`. Also used by `colorSample.ts`. */
export const blendToComposite = (mode: CanvasBlendMode): GlobalCompositeOperation =>
  mode === 'normal' ? 'source-over' : (mode as GlobalCompositeOperation);

const setTransformFromMat = (ctx: Ctx, m: Mat2d): void => {
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
};

const identityTransform = (ctx: Ctx): void => {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
};

/**
 * Builds a small, reusable two-tone checkerboard tile (a 2×2 grid of `squarePx`
 * cells) through the {@link RasterBackend} seam. The engine creates this ONCE and
 * feeds it back via {@link CompositeOptions.checkerboardTile}; each frame then
 * only `createPattern`s over it, so no per-cell fill loop runs per frame.
 */
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
 * Fills the ENTIRE viewport with the checkerboard pattern. The canvas is a
 * virtually infinite plane (like legacy): the checker IS the world surround, not
 * a document backdrop — the document rect is no longer a visual boundary, so the
 * contract's `background` field no longer renders. When no tile is provided the
 * checkerboard is off and the cleared surface shows the widget's `bg.inset`.
 *
 * The pattern is laid with the identity transform in place, anchoring its cells
 * to the screen (canvas) origin at a fixed pixel size — like legacy, which pins
 * the pattern to the stage — so it stays visually stable and never swims or
 * scales while panning/zooming.
 */
const drawBackground = (ctx: Ctx, target: RasterSurface, tile: RasterSurface | null): void => {
  if (!tile) {
    // Checkerboard disabled: leave the cleared surface showing `bg.inset`.
    return;
  }
  const pattern = ctx.createPattern(tile.canvas, 'repeat');
  if (!pattern) {
    return;
  }
  ctx.fillStyle = pattern;
  ctx.fillRect(0, 0, target.width, target.height);
};

const isMaskLayer = (
  layer: CanvasLayerContract
): layer is Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }> =>
  layer.type === 'regional_guidance' || layer.type === 'inpaint_mask';

/**
 * Draws a mask-bearing layer as a TINTED, TRANSLUCENT overlay. With a backend
 * available (the engine path) it colorizes the mask's alpha stencil with the
 * layer's fill colour/pattern (`source-in`) on an intermediate surface, then
 * blits that through the current (already transformed, `globalAlpha =
 * layer.opacity`) context — legacy's `source-in` compositing-rect technique.
 * Without a backend it falls back to blitting the coverage plus a flat
 * translucent fill.
 */
const drawMaskLayer = (
  ctx: Ctx,
  layer: Extract<CanvasLayerContract, { type: 'regional_guidance' | 'inpaint_mask' }>,
  surface: RasterSurface,
  origin: Vec2,
  opts: CompositeOptions
): void => {
  const fill = layer.mask.fill;
  if (opts.backend) {
    const tile = opts.maskPatternTile ? opts.maskPatternTile(fill.style, fill.color) : null;
    const colorized = colorizeMask(opts.backend, surface, surface.width, surface.height, fill, tile);
    // The outer loop already set globalAlpha = layer.opacity; blit the colorized
    // overlay at the mask's local content origin.
    ctx.drawImage(colorized.canvas, origin.x, origin.y);
    return;
  }
  // Backend-less fallback (bare test call): coverage + flat translucent fill.
  ctx.drawImage(surface.canvas, origin.x, origin.y);
  ctx.globalAlpha = layer.opacity * MASK_TINT_ALPHA;
  ctx.fillStyle = fill.color;
  ctx.fillRect(origin.x, origin.y, surface.width, surface.height);
};

/**
 * Draws one cached layer through its transform, applying opacity/blend and any
 * transient transform override. Mask-bearing layers are colorized; everything
 * else is a straight blit of its content-sized cache surface.
 */
const drawCachedLayer = (
  ctx: Ctx,
  layer: CanvasLayerContract,
  entry: LayerCacheEntry,
  view: Mat2d,
  opts: CompositeOptions
): void => {
  ctx.save();
  ctx.globalAlpha = layer.opacity;
  ctx.globalCompositeOperation = blendToComposite(layer.blendMode);

  const override = opts.transformOverrides?.get(layer.id);
  const layerMat = fromTRS(
    { x: override?.x ?? layer.transform.x, y: override?.y ?? layer.transform.y },
    override?.rotation ?? layer.transform.rotation,
    override?.scaleX ?? layer.transform.scaleX,
    override?.scaleY ?? layer.transform.scaleY
  );
  setTransformFromMat(ctx, multiply(view, layerMat));

  // The cache surface holds pixels for `entry.rect` in layer-local space; draw
  // it at that local origin (offset paint/mask layers place their content off-zero).
  const origin = { x: entry.rect.x, y: entry.rect.y };
  const preview = opts.onlyLayerId ? null : (opts.layerPreviews?.get(layer.id) ?? null);
  if (preview) {
    // Non-destructive filter preview: draw the full backend output at its
    // layer-local rect (through the already-applied layer transform), including
    // the same display-only control transparency effect used after commit.
    const displayPreview =
      layer.type === 'control' && layer.withTransparencyEffect && opts.backend
        ? renderControlTransparency(opts.backend, preview.surface, preview.surface.width, preview.surface.height)
        : preview.surface;
    ctx.drawImage(displayPreview.canvas, preview.rect.x, preview.rect.y);
  } else if (isMaskLayer(layer)) {
    drawMaskLayer(ctx, layer, entry.surface, origin, opts);
  } else if (layer.type === 'control' && layer.withTransparencyEffect && opts.backend) {
    // Display-only lightness→alpha effect (legacy `LightnessToAlphaFilter`): dark
    // areas of the control map drop out so underlying content shows through.
    const effect = renderControlTransparency(opts.backend, entry.surface, entry.surface.width, entry.surface.height);
    ctx.drawImage(effect.canvas, origin.x, origin.y);
  } else {
    // Raster layers may carry non-destructive adjustments; the engine supplies a
    // memoized adjusted surface (never recomputed per frame). Fall back to the raw
    // cache when there are no adjustments or no provider.
    const adjusted = layer.type === 'raster' && opts.adjustedSurface ? opts.adjustedSurface(layer, entry) : null;
    ctx.drawImage((adjusted ?? entry.surface).canvas, origin.x, origin.y);
  }

  ctx.restore();
};

/**
 * Composites `doc` onto `target`, using `caches` for each layer's pixels and
 * `view` as the document→screen transform.
 */
export const compositeDocument = (
  target: RasterSurface,
  doc: CanvasDocumentContractV2,
  caches: LayerCacheStore,
  view: Mat2d,
  opts: CompositeOptions = {}
): void => {
  const ctx = target.ctx;

  ctx.save();

  // Smoothing policy for all layer/staged `drawImage` blits below. Set once
  // under the outer save (inner per-layer save/restore preserves it). Off when
  // zoomed in keeps magnified pixels crisp and skips the costly bilinear upscale.
  ctx.imageSmoothingEnabled = opts.imageSmoothing ?? true;

  // Clear the whole target in screen space, then lay the checkerboard across the
  // entire viewport — the canvas is an unbounded plane, so the checker is the
  // world, not a document backdrop. With the checkerboard off the cleared surface
  // shows the widget's themed `bg.inset` through it.
  identityTransform(ctx);
  ctx.clearRect(0, 0, target.width, target.height);
  drawBackground(ctx, target, opts.checkerboardTile ?? null);

  // Composite in STRICT GROUP ORDER (legacy `arrangeEntities` / `CanvasEntityRendererModule`):
  // raster (bottom) < control < regional guidance < inpaint mask (top). This
  // matches the layers panel, which renders these as fixed grouped sections, and
  // makes a layer's global insertion index irrelevant ACROSS groups (a raster
  // created after a control layer must still draw below it). WITHIN a group the
  // panel/array relative order is preserved. The `stagedPreview` lands on top of
  // all groups below. Index 0 is top-most, so iterate the array in reverse.
  const drawGroup = (rank: number): void => {
    for (let i = doc.layers.length - 1; i >= 0; i--) {
      const layer = doc.layers[i];
      if (
        !layer ||
        (!layer.isEnabled && layer.id !== opts.onlyLayerId) ||
        layer.id === opts.skipLayerId ||
        (opts.onlyLayerId && layer.id !== opts.onlyLayerId) ||
        layerGroupRank(layer) !== rank
      ) {
        continue;
      }
      const entry = caches.get(layer.id);
      // Skip layers with no cache or an empty content rect (a brand-new / cleared
      // paint / mask layer holds no pixels — nothing to draw).
      if (!entry || entry.rect.width <= 0 || entry.rect.height <= 0) {
        continue;
      }
      drawCachedLayer(ctx, layer, entry, view, opts);
    }
  };
  for (let rank = 0; rank < LAYER_GROUP_COUNT; rank++) {
    drawGroup(rank);
  }

  // Staged generation preview over its bbox (document space), with a subtle
  // dashed outline so the pending result reads distinctly from committed pixels.
  const staged = opts.onlyLayerId ? null : opts.stagedPreview;
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
