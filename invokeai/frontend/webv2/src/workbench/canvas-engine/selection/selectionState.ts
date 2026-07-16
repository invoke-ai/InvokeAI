/**
 * Engine-transient pixel selection.
 *
 * Per the plan's state-tier table, a selection is *interaction* state: it lives
 * on the engine, never in the reducer contract, and is not undoable. The source
 * of truth is the set of closed `Path2D` polygons the lasso tool commits; the
 * derived artifact is a bounded **mask surface** (alpha 255 inside the
 * selection), sized to the selection extent and placed in document space (a
 * {@link PlacedSurface}), built through the {@link RasterBackend} seam. Boolean ops
 * (`replace`/`add`/`subtract`/`intersect`) are applied to the MASK — that is the
 * single source of truth for painting/fill clipping. The committed path list is
 * kept only for {@link marchingAnts} rendering (stroking the outlines), not for
 * re-deriving the mask.
 *
 * ## Marching-ants approximation
 *
 * Ants are stroked from the committed path LIST (each path drawn dashed), not by
 * tracing the mask's true edge. This matches the mask exactly for `replace` and
 * `add`, and reads correctly for `subtract` (the cut-out path's outline shows as
 * a hole) and `selectAll`/`invert` (a document-border path is included). It is an
 * approximation for `intersect` and for heavily overlapping compositions, where
 * the true selection outline is the boolean result rather than the union of the
 * source outlines — the ants may show interior source edges that the mask does
 * not. The exception is {@link SelectionState.replaceMask} (pixel-mask
 * replacement, e.g. a Select Object result), whose ants ARE traced from the
 * mask's true edge via {@link traceMaskOutlinePath}. The mask itself is always
 * exact.
 *
 * ## Emptiness
 *
 * `hasSelection` is tracked structurally, not by scanning pixels (a scan is
 * unavailable on the node raster stub and costly on the DOM). Consequently,
 * `subtract`ing away the entire selection leaves `hasSelection` true until an
 * explicit deselect — a known, documented limitation of this phase.
 *
 * Zero React, zero import-time side effects.
 */

import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PlacedSurface, Rect, SelectionOp, Vec2 } from '@workbench/canvas-engine/types';

import { intersect, isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';
import { traceMaskOutlinePath } from '@workbench/canvas-engine/selection/maskOutline';

/** A committed selection contribution: the closed path and the op it applied. */
export interface SelectionCommit {
  path: Path2D;
  /** The path's document-space bounds (used to maintain the selection bounds cheaply). */
  bounds: Rect;
  op: SelectionOp;
}

/** The engine-facing selection handle. */
export interface SelectionState {
  /** Whether a selection currently exists (drives clip/fill/ants gating). */
  hasSelection(): boolean;
  /** The selection's document-space bounds, or `null` when empty. */
  bounds(): Rect | null;
  /**
   * The selection mask as a placed surface (alpha 255 inside) in document space,
   * bounded to the selection extent (the `rect` records its origin/size), or
   * `null` when empty.
   */
  mask(): PlacedSurface | null;
  /** The committed path outlines, for marching-ants rendering. */
  antsPaths(): readonly Path2D[];
  /** True if `p` (document space) is inside the selection. Cheap 1×1 read; `false` when empty. */
  containsPoint(p: Vec2): boolean;
  /** Applies a committed lasso path against the mask with the given boolean op. */
  commit(commit: SelectionCommit): void;
  /** Replaces the selection with an alpha-bearing surface placed in document space. */
  replaceMask(mask: PlacedSurface): void;
  /** Selects the whole `domain` rectangle (the engine passes `content ∪ bbox`). */
  selectAll(domain: Rect): void;
  /**
   * Inverts the selection within `domain` (empty → select `domain`). The domain
   * is the bounded region the complement is taken over — the engine passes
   * `content ∪ bbox`, the coherent analogue of legacy's bounded canvas now that
   * the document rect is retired.
   */
  invert(domain: Rect): void;
  /** Clears the selection (deselect). */
  clear(): void;
  /** Releases the mask surface reference. */
  dispose(): void;
}

/** Dependencies injected by the engine. */
export interface SelectionStateDeps {
  backend: RasterBackend;
  createPath2D: CreatePath2D;
  /** Current document pixel size (the mask surface size), or `null` when no document. */
  getDocumentSize(): { width: number; height: number } | null;
  /** Called after every mutation (engine syncs the `hasSelection` store + overlay/ants). */
  onChange(): void;
}

const MASK_FILL = '#ffffff';

/** SVG path data for a closed rectangle in document space. */
const rectToPathData = (r: Rect): string =>
  `M ${r.x} ${r.y} L ${r.x + r.width} ${r.y} L ${r.x + r.width} ${r.y + r.height} L ${r.x} ${r.y + r.height} Z`;

/** Builds a closed rectangle `Path2D` (document space) via the injected factory. */
const rectPath = (createPath2D: CreatePath2D, r: Rect): Path2D => createPath2D(rectToPathData(r));

/** The canvas composite op that realizes each boolean selection op on the mask. */
const compositeForOp = (op: SelectionOp): GlobalCompositeOperation => {
  switch (op) {
    case 'add':
    case 'replace':
      return 'source-over';
    case 'subtract':
      return 'destination-out';
    case 'intersect':
      return 'destination-in';
  }
};

/** Creates an engine-transient selection bound to the raster backend seam. */
export const createSelectionState = (deps: SelectionStateDeps): SelectionState => {
  const { backend, createPath2D, getDocumentSize, onChange } = deps;

  let mask: RasterSurface | null = null;
  // The mask surface's document-space bounds (its origin/extent). The mask is
  // bounded to the selection, not the document — surface pixel (sx,sy) maps to
  // document (maskRect.x+sx, maskRect.y+sy).
  let maskRect: Rect | null = null;
  let commits: SelectionCommit[] = [];
  let selectionBounds: Rect | null = null;
  let selected = false;

  /**
   * Ensures the mask surface exists and covers `rect` (integer bounds),
   * preserving any existing mask pixels at their new offset when it must grow.
   */
  const ensureMask = (rect: Rect): RasterSurface => {
    const want: Rect = roundOut(rect);
    if (!mask || !maskRect) {
      mask = backend.createSurface(Math.max(0, want.width), Math.max(0, want.height));
      maskRect = want;
      return mask;
    }
    const grown = union(maskRect, want);
    if (
      grown.x === maskRect.x &&
      grown.y === maskRect.y &&
      grown.width === maskRect.width &&
      grown.height === maskRect.height
    ) {
      return mask;
    }
    // Grow, preserving old pixels at their new offset (snapshot before resize).
    let snapshot: ImageData | null = null;
    if (maskRect.width > 0 && maskRect.height > 0) {
      snapshot = mask.ctx.getImageData(0, 0, maskRect.width, maskRect.height);
    }
    mask.resize(grown.width, grown.height);
    if (snapshot) {
      mask.ctx.putImageData(snapshot, maskRect.x - grown.x, maskRect.y - grown.y);
    }
    maskRect = grown;
    return mask;
  };

  /** Resets the mask surface to exactly `rect` (cleared), for `replace`-style ops. */
  const resetMask = (rect: Rect): RasterSurface => {
    const want: Rect = roundOut(rect);
    if (!mask) {
      mask = backend.createSurface(Math.max(0, want.width), Math.max(0, want.height));
    } else if (mask.width !== want.width || mask.height !== want.height) {
      mask.resize(Math.max(0, want.width), Math.max(0, want.height));
    }
    maskRect = want;
    clearSurface(mask);
    return mask;
  };

  /**
   * Fills a path into the mask under `op`'s composite mode. The mask surface is
   * offset by `maskRect.origin`, so the (document-space) path is drawn through a
   * translate that maps document → surface coordinates.
   */
  const fillPath = (surface: RasterSurface, origin: Vec2, path: Path2D, op: GlobalCompositeOperation): void => {
    const ctx = surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, -origin.x, -origin.y);
    ctx.globalCompositeOperation = op;
    ctx.globalAlpha = 1;
    ctx.fillStyle = MASK_FILL;
    ctx.fill(path);
    ctx.restore();
  };

  const clearSurface = (surface: RasterSurface): void => {
    const ctx = surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, surface.width, surface.height);
    ctx.restore();
  };

  const clear = (): void => {
    commits = [];
    selectionBounds = null;
    selected = false;
    if (mask) {
      clearSurface(mask);
    }
    onChange();
  };

  const replaceMask = (next: PlacedSurface): void => {
    const rect = roundOut(next.rect);
    const publishEmptyReplacement = (): void => {
      // Replacing from a valid empty alpha result should not clear the old
      // surface in place: clearSurface() is fallible and would run after the
      // logical selection flags had already changed. Detach the old surface and
      // atomically publish an empty selection instead; it is reclaimed by GC.
      mask = null;
      maskRect = null;
      commits = [];
      selectionBounds = null;
      selected = false;
      onChange();
    };
    if (isEmpty(rect) || next.surface.width <= 0 || next.surface.height <= 0) {
      publishEmptyReplacement();
      return;
    }

    const source = next.surface.ctx.getImageData(0, 0, next.surface.width, next.surface.height);
    let hasAlpha = false;
    for (let index = 3; index < source.data.length; index += 4) {
      if (source.data[index] !== 0) {
        hasAlpha = true;
        break;
      }
    }
    if (!hasAlpha) {
      publishEmptyReplacement();
      return;
    }

    const copiedData = new Uint8ClampedArray(source.data);
    const copied =
      typeof ImageData === 'undefined'
        ? ({ colorSpace: source.colorSpace, data: copiedData, height: source.height, width: source.width } as ImageData)
        : new ImageData(copiedData, source.width, source.height);
    // Prepare every fallible replacement artifact before publishing any state.
    // If allocation, pixel upload, or Path2D construction fails, the exact prior
    // selection remains authoritative and the engine may safely report failure.
    const nextMask = backend.createSurface(rect.width, rect.height);
    nextMask.ctx.putImageData(copied, 0, 0);
    // Trace the mask's true edge for the ants; a mask whose alpha never reaches
    // the solid threshold still selected something (hasAlpha above), so fall
    // back to tracing any non-zero coverage rather than showing no outline.
    const alphaSource = { data: copiedData, height: source.height, width: source.width };
    const outline =
      traceMaskOutlinePath(alphaSource, rect) || traceMaskOutlinePath(alphaSource, rect, 1) || rectToPathData(rect);
    const nextPath = createPath2D(outline);

    mask = nextMask;
    maskRect = rect;
    commits = [{ bounds: rect, op: 'replace', path: nextPath }];
    selectionBounds = rect;
    selected = true;
    onChange();
  };

  const commit = (next: SelectionCommit): void => {
    if (!getDocumentSize()) {
      // No active document ⇒ no selection surface to build.
      return;
    }
    const bounds = isEmpty(next.bounds) ? null : roundOut(next.bounds);

    if (next.op === 'replace') {
      if (!bounds) {
        clear();
        return;
      }
      const surface = resetMask(bounds);
      fillPath(surface, { x: maskRect!.x, y: maskRect!.y }, next.path, 'source-over');
      commits = [next];
      selectionBounds = bounds;
      selected = true;
      onChange();
      return;
    }

    if (next.op === 'intersect' && (!selected || !selectionBounds)) {
      // Intersecting with nothing yields nothing.
      clear();
      return;
    }

    // For `add` the mask may need to grow to include the new path; `subtract` /
    // `intersect` never grow the covered region (they only remove).
    const surface =
      next.op === 'add' && bounds
        ? ensureMask(bounds)
        : ensureMask(selectionBounds ?? bounds ?? { height: 0, width: 0, x: 0, y: 0 });
    fillPath(surface, { x: maskRect!.x, y: maskRect!.y }, next.path, compositeForOp(next.op));
    commits.push(next);

    switch (next.op) {
      case 'add':
        selectionBounds = selectionBounds && bounds ? union(selectionBounds, bounds) : (bounds ?? selectionBounds);
        selected = selected || bounds !== null;
        break;
      case 'subtract':
        // Bounds can only shrink; without a pixel scan we keep the (over-approx)
        // prior bounds. `selected` stays true — see module docs.
        break;
      case 'intersect':
        selectionBounds = selectionBounds && bounds ? intersect(selectionBounds, bounds) : null;
        selected = selectionBounds !== null;
        break;
    }
    onChange();
  };

  const selectAll = (domain: Rect): void => {
    const rect = roundOut(domain);
    const surface = resetMask(rect);
    const ctx = surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = MASK_FILL;
    ctx.fillRect(0, 0, surface.width, surface.height);
    ctx.restore();
    commits = [{ bounds: rect, op: 'replace', path: rectPath(createPath2D, rect) }];
    selectionBounds = rect;
    selected = !isEmpty(rect);
    onChange();
  };

  const invert = (domain: Rect): void => {
    if (!selected || !mask || !maskRect) {
      // Inverting an empty selection selects the whole domain.
      selectAll(domain);
      return;
    }
    const rect = roundOut(domain);
    // The current mask, captured over the domain: draw the (offset) mask onto a
    // domain-sized temp so the complement is taken in domain-local coordinates.
    const temp = backend.createSurface(Math.max(0, rect.width), Math.max(0, rect.height));
    const tctx = temp.ctx;
    tctx.setTransform(1, 0, 0, 1, 0, 0);
    tctx.globalCompositeOperation = 'source-over';
    tctx.globalAlpha = 1;
    tctx.fillStyle = MASK_FILL;
    tctx.fillRect(0, 0, temp.width, temp.height);
    tctx.globalCompositeOperation = 'destination-out';
    // Punch out the current mask at its offset within the domain.
    tctx.drawImage(mask.canvas, maskRect.x - rect.x, maskRect.y - rect.y);

    const surface = resetMask(rect);
    const ctx = surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(temp.canvas, 0, 0);
    ctx.restore();

    // Ants: the outer domain border plus the former outlines (now holes).
    commits = [{ bounds: rect, op: 'replace', path: rectPath(createPath2D, rect) }, ...commits];
    selectionBounds = rect;
    selected = true;
    onChange();
  };

  const containsPoint = (p: Vec2): boolean => {
    if (!selected || !mask || !maskRect) {
      return false;
    }
    const x = Math.floor(p.x) - maskRect.x;
    const y = Math.floor(p.y) - maskRect.y;
    if (x < 0 || y < 0 || x >= mask.width || y >= mask.height) {
      return false;
    }
    const pixel = mask.ctx.getImageData(x, y, 1, 1);
    return pixel.data[3] !== undefined && pixel.data[3] > 0;
  };

  return {
    antsPaths: () => commits.map((entry) => entry.path),
    bounds: () => selectionBounds,
    clear,
    commit,
    containsPoint,
    dispose: () => {
      mask = null;
      maskRect = null;
      commits = [];
      selectionBounds = null;
      selected = false;
    },
    hasSelection: () => selected,
    invert,
    mask: () => (selected && mask && maskRect ? { rect: maskRect, surface: mask } : null),
    replaceMask,
    selectAll,
  };
};
