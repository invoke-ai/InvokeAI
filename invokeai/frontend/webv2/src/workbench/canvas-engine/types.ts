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

/** Canvas/DOMMatrix affine order: x' = a*x + c*y + e; y' = b*x + d*y + f. */
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
  | 'marquee'
  | 'move'
  | 'transform'
  | 'bbox'
  | 'view'
  | 'colorPicker'
  | 'text'
  | 'sam';

/** Selection operations replace, union, subtract or intersect the existing alpha mask. */
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

/** Next-frame render requirements coalesce in the scheduler to avoid unnecessary full repaint. */
export interface RenderFlags {
  /** The viewport transform (pan/zoom) changed. */
  view: boolean;
  /** Ids of layers whose pixel content or transform changed. */
  layers: Set<string>;
  /** Interaction overlays (selection, cursors, guides) changed. */
  overlay: boolean;
  /** Force a full repaint, ignoring the other flags. */
  all: boolean;
  /** What of the composite must be repainted; any invalidation without a region widens it to full. */
  damage: FrameDamage;
}

/**
 * Composite repaint extent: everything, nothing visible, or the screen bounds of layer-local regions. Regions that
 * resolve offscreen repaint nothing.
 */
export type FrameDamage =
  | { readonly kind: 'full' }
  | { readonly kind: 'none' }
  | { readonly kind: 'regions'; readonly regions: readonly LayerDamage[] };

export const FULL_DAMAGE: FrameDamage = { kind: 'full' };
export const NO_DAMAGE: FrameDamage = { kind: 'none' };

/** Changed region in layer-local coordinates; compositor applies the effective layer matrix to reach screen space. */
export interface LayerDamage {
  layerId: string;
  rect: Rect;
}
