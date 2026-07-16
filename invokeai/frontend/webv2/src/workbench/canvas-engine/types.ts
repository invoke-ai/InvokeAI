/**
 * Core types for the canvas engine.
 *
 * This module is the shared vocabulary for the imperative canvas engine
 * (`src/workbench/canvas-engine/`). It has zero React imports and zero
 * side effects at import time — it is pure type/interface declarations (the
 * single type-only import below is erased at compile time).
 */

import type { RasterSurface } from './render/raster';

/** A 2D point or vector. */
export interface Vec2 {
  x: number;
  y: number;
}

/** An axis-aligned rectangle. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * A raster surface placed in some coordinate space: the surface holds pixels for
 * `rect` (its bounds in that space), so surface pixel `(sx, sy)` maps to
 * `(rect.x + sx, rect.y + sy)`. Used for content-sized layer caches (layer-local
 * space) and the bounded selection mask (document space).
 */
export interface PlacedSurface {
  surface: RasterSurface;
  rect: Rect;
}

/**
 * A 2D affine transform matrix, using the canvas transform convention:
 *
 * ```
 * x' = a*x + c*y + e
 * y' = b*x + d*y + f
 * ```
 *
 * This matches `CanvasRenderingContext2D.setTransform(a, b, c, d, e, f)` and
 * the DOMMatrix 2D constructor argument order.
 */
export interface Mat2d {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

/** Identifiers for the tools the engine's interaction layer can dispatch to. */
export type ToolId =
  | 'brush'
  | 'eraser'
  | 'shape'
  | 'gradient'
  | 'lasso'
  | 'move'
  | 'transform'
  | 'bbox'
  | 'view'
  | 'colorPicker'
  | 'text'
  | 'sam';

/**
 * A pixel-selection boolean operation applied when a new lasso path commits
 * against the existing selection mask (see `selection/selectionState.ts`):
 * `replace` swaps it, `add` unions, `subtract` cuts out, `intersect` keeps the
 * overlap.
 */
export type SelectionOp = 'replace' | 'add' | 'subtract' | 'intersect';

/** Modifier key state accompanying a pointer sample. */
export interface PointerModifiers {
  shift: boolean;
  alt: boolean;
  ctrl: boolean;
  meta: boolean;
}

/** A normalized pointer sample fed into the engine's interaction layer. */
export interface PointerInput {
  /** Pointer position in document space (unaffected by pan/zoom). */
  documentPoint: Vec2;
  /** Pointer position in screen/viewport space (CSS pixels). */
  screenPoint: Vec2;
  /** Pressure in [0, 1]. 0.5 is used for devices that don't report pressure. */
  pressure: number;
  /** Bitmask of currently-pressed buttons, matching `PointerEvent.buttons`. */
  buttons: number;
  modifiers: PointerModifiers;
  pointerType: 'mouse' | 'pen' | 'touch';
  /** High-resolution timestamp (ms), matching `PointerEvent.timeStamp`. */
  timeStamp: number;
}

/**
 * Describes what needs to be re-rendered on the next frame. Consumers
 * (scheduler/compositor, added in later tasks) union flags together rather
 * than always doing a full repaint.
 */
export interface RenderFlags {
  /** The viewport transform (pan/zoom) changed. */
  view: boolean;
  /** Ids of layers whose pixel content or transform changed. */
  layers: Set<string>;
  /** Interaction overlays (selection, cursors, guides) changed. */
  overlay: boolean;
  /** Force a full repaint, ignoring the other flags. */
  all: boolean;
}
