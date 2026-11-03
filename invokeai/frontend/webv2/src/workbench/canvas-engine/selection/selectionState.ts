/**
 * Engine-owned, unpersisted selection state uses snapshots for undo. The bounded document-space alpha mask is
 * authoritative for clipping and boolean operations; paths serve only ants rendering.
 *
 * Replace uses its path; boolean operations retrace actual mask edges. Add may fall back to source paths when stub
 * pixels cannot be read; empty subtract/intersect results clear selection. Replace/add emptiness is structural,
 * while subtract/intersect inspect alpha.
 */

import type { CreatePath2D } from '@workbench/canvas-engine/freehand';
import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { PlacedSurface, Rect, SelectionOp, Vec2 } from '@workbench/canvas-engine/types';

import { intersect, isEmpty, roundOut, union } from '@workbench/canvas-engine/math/rect';
import { traceMaskOutlinePath } from '@workbench/canvas-engine/selection/maskOutline';
import { rectPathData } from '@workbench/canvas-engine/selection/selectionPaths';

/** A committed selection contribution: the closed path and the op it applied. */
export interface SelectionCommit {
  path: Path2D;
  /** The path's document-space bounds (used to maintain the selection bounds cheaply). */
  bounds: Rect;
  op: SelectionOp;
}

/**
 * A restorable capture of the selection: the mask's alpha plane over `rect`
 * plus the bookkeeping the mask alone cannot recover. Alpha-only keeps a
 * full-document capture at one byte per pixel; the mask is white by contract.
 */
export interface SelectionSnapshot {
  readonly alpha: Uint8ClampedArray<ArrayBuffer> | null;
  readonly rect: Rect | null;
  readonly commits: readonly SelectionCommit[];
  readonly bounds: Rect | null;
  readonly selected: boolean;
}

/** The engine-facing selection handle. */
export interface SelectionState {
  /** Whether a selection currently exists (drives clip/fill/ants gating). */
  hasSelection(): boolean;
  /** The selection's document-space bounds, or `null` when empty. */
  bounds(): Rect | null;
  /** Document-space mask surface and extent, or null when empty. */
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
  /** Complements within a bounded domain; empty selection selects it. Engine domain is content union bbox. */
  invert(domain: Rect): void;
  /** Clears the selection (deselect). */
  clear(): void;
  /** Captures the selection for a later {@link restore}; the capture is immutable and detached. */
  snapshot(): SelectionSnapshot;
  /** Reinstates a captured selection exactly, notifying like any other mutation. */
  restore(snapshot: SelectionSnapshot): void;
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

/** `ImageData` is absent on the node raster stub; a structural stand-in carries the same fields. */
const createImageData = (data: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): ImageData =>
  typeof ImageData === 'undefined'
    ? ({ colorSpace: 'srgb', data, height, width } as ImageData)
    : new ImageData(data, width, height);

const alphaOf = (pixels: ImageData): Uint8ClampedArray<ArrayBuffer> => {
  const alpha = new Uint8ClampedArray(pixels.width * pixels.height);
  for (let index = 0; index < alpha.length; index += 1) {
    alpha[index] = pixels.data[index * 4 + 3] ?? 0;
  }
  return alpha;
};

/** Expands an alpha plane back into the mask's white-with-coverage pixels. */
const pixelsOf = (alpha: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): ImageData => {
  const data = new Uint8ClampedArray(alpha.length * 4).fill(255);
  for (let index = 0; index < alpha.length; index += 1) {
    data[index * 4 + 3] = alpha[index] ?? 0;
  }
  return createImageData(data, width, height);
};

/** Builds a closed rectangle `Path2D` (document space) via the injected factory. */
const rectPath = (createPath2D: CreatePath2D, r: Rect): Path2D => createPath2D(rectPathData(r));

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
  // The last capture stays valid until the next mutation, so a change recorded
  // as before/after reads the mask back once, not twice.
  let cachedSnapshot: SelectionSnapshot | null = null;

  const changed = (): void => {
    cachedSnapshot = null;
    onChange();
  };

  /** Ensure integer mask bounds, preserving old pixels at their shifted offset when growing. */
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

  /** Draw document paths with document-to-mask translation under the boolean operation's composite mode. */
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
    changed();
  };

  const replaceMask = (next: PlacedSurface): void => {
    const rect = roundOut(next.rect);
    const publishEmptyReplacement = (): void => {
      // Publish empty selection by detaching the old surface; a fallible in-place clear after changing logical
      // flags would break atomicity.
      mask = null;
      maskRect = null;
      commits = [];
      selectionBounds = null;
      selected = false;
      changed();
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

    // Prepare every fallible replacement artifact before publishing any state.
    // If allocation, the blit, or Path2D construction fails, the exact prior
    // selection remains authoritative and the engine may safely report failure.
    const nextMask = backend.createSurface(rect.width, rect.height);
    nextMask.ctx.drawImage(next.surface.canvas, 0, 0);
    // Trace the mask's true edge for the ants; a mask whose alpha never reaches
    // the solid threshold still selected something (hasAlpha above), so fall
    // back to tracing any non-zero coverage rather than showing no outline.
    const outline = traceMaskOutlinePath(source, rect) || traceMaskOutlinePath(source, rect, 1) || rectPathData(rect);
    const nextPath = createPath2D(outline);

    mask = nextMask;
    maskRect = rect;
    commits = [{ bounds: rect, op: 'replace', path: nextPath }];
    selectionBounds = rect;
    selected = true;
    changed();
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
      changed();
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

    switch (next.op) {
      case 'add':
        selectionBounds = selectionBounds && bounds ? union(selectionBounds, bounds) : (bounds ?? selectionBounds);
        selected = selected || bounds !== null;
        break;
      case 'subtract':
        // Retain conservative prior bounds; tracing determines whether coverage remains.
        break;
      case 'intersect':
        selectionBounds = selectionBounds && bounds ? intersect(selectionBounds, bounds) : null;
        selected = selectionBounds !== null;
        break;
    }

    const outline = selected ? traceCurrentOutline() : null;
    if (outline) {
      commits = [{ bounds: selectionBounds ?? roundOut(next.bounds), op: 'replace', path: createPath2D(outline) }];
    } else if (next.op === 'add') {
      commits.push(next);
    } else {
      clear();
      return;
    }
    changed();
  };

  /** The mask's true edge as path data, or null when it holds no pixels. */
  const traceCurrentOutline = (): string | null => {
    if (!mask || !maskRect || isEmpty(maskRect)) {
      return null;
    }
    const pixels = mask.ctx.getImageData(0, 0, maskRect.width, maskRect.height);
    // One alpha scan detects emptied subtraction without two full traces.
    let hasAlpha = false;
    for (let index = 3; index < pixels.data.length; index += 4) {
      if (pixels.data[index] !== 0) {
        hasAlpha = true;
        break;
      }
    }
    if (!hasAlpha) {
      return null;
    }
    const source = { data: pixels.data, height: maskRect.height, width: maskRect.width };
    return traceMaskOutlinePath(source, maskRect) || traceMaskOutlinePath(source, maskRect, 1) || null;
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
    changed();
  };

  const invert = (domain: Rect): void => {
    if (!selected || !mask || !maskRect) {
      // Inverting an empty selection selects the whole domain.
      selectAll(domain);
      return;
    }
    const rect = roundOut(domain);
    // Copy the placed mask into a domain-local temporary before complementing.
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
    changed();
  };

  const snapshot = (): SelectionSnapshot => {
    if (cachedSnapshot) {
      return cachedSnapshot;
    }
    const captured = selected && mask && maskRect && !isEmpty(maskRect) ? maskRect : null;
    cachedSnapshot = {
      alpha: captured ? alphaOf(mask!.ctx.getImageData(0, 0, captured.width, captured.height)) : null,
      bounds: selectionBounds,
      commits: [...commits],
      rect: captured,
      selected,
    };
    return cachedSnapshot;
  };

  const restore = (next: SelectionSnapshot): void => {
    if (next.alpha && next.rect) {
      const surface = resetMask(next.rect);
      surface.ctx.putImageData(pixelsOf(next.alpha, next.rect.width, next.rect.height), 0, 0);
    } else if (mask) {
      clearSurface(mask);
    }
    commits = [...next.commits];
    selectionBounds = next.bounds;
    selected = next.selected;
    cachedSnapshot = next;
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
      cachedSnapshot = null;
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
    restore,
    selectAll,
    snapshot,
  };
};
