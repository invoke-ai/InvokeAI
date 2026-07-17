/**
 * The pan/zoom viewport: the single source of truth for how document space
 * maps to the on-screen canvas. It owns `{ pan, zoom }`, the CSS viewport
 * size, and the device-pixel ratio, and derives the document→screen
 * transform used by both the renderer (device pixels) and pointer routing
 * (CSS pixels).
 *
 * ## Coordinate convention
 *
 * - **Document space**: the canvas document's own pixel grid, unaffected by
 *   pan/zoom (a layer at document `(0,0)` is the top-left of the document).
 * - **Screen/CSS space**: CSS pixels relative to the canvas element's
 *   top-left. This is what `PointerEvent.clientX/Y` (minus the element rect)
 *   gives you. `pan` is stored in CSS pixels.
 * - **Device space**: the canvas backing store, `CSS × dpr`. The renderer
 *   draws here.
 *
 * The mappings are:
 * ```
 * screen = zoom * doc + pan            (CSS pixels)
 * device = dpr * screen = (zoom*dpr) * doc + dpr*pan
 * ```
 * so `viewMatrix(dpr)` is `scale(zoom*dpr)` followed by a `dpr*pan`
 * translation — i.e. `a = d = zoom*dpr`, `e = dpr*pan.x`, `f = dpr*pan.y`.
 *
 * Zero React, zero import-time side effects; DOM-free (operates on plain
 * numbers), so it runs unchanged in node tests.
 */

import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, invert } from '@workbench/canvas-engine/math/mat2d';
import { clampZoom, snapZoom } from '@workbench/canvas-engine/math/snapping';
import {
  panBy as calculatePanBy,
  wheelZoomAtPoint as calculateWheelZoomAtPoint,
  zoomAtPoint as calculateZoomAtPoint,
} from '@workbench/panZoom';

export { WHEEL_ZOOM_STEP } from '@workbench/panZoom';

/** Maximum device-pixel ratio the engine honors; higher DPRs are clamped to bound backing-store size. */
export const MAX_DPR = 2;

/** The mutable view state. */
export interface ViewState {
  pan: Vec2;
  zoom: number;
}

/** A `{ width, height }` size in CSS pixels. */
export interface Size {
  width: number;
  height: number;
}

/** The imperative viewport handle. */
export interface Viewport {
  /** Current `{ pan, zoom }` (a fresh copy — never mutate the return value). */
  getState(): ViewState;
  /** Current zoom factor. */
  getZoom(): number;
  /** Current CSS viewport size. */
  getViewportSize(): Size;
  /** Current (clamped) device-pixel ratio. */
  getDpr(): number;
  /** The document→device-pixel transform for the given `dpr`. */
  viewMatrix(dpr: number): Mat2d;
  /** Maps a CSS-pixel screen point to document space. */
  screenToDocument(p: Vec2): Vec2;
  /** Maps a document point to a CSS-pixel screen point. */
  documentToScreen(p: Vec2): Vec2;
  /** Sets zoom while keeping the document point under `screenAnchor` fixed. */
  zoomAtPoint(newZoom: number, screenAnchor: Vec2): void;
  /** Exponential wheel zoom about `screenAnchor`, snapping near common zoom levels. */
  wheelZoom(deltaY: number, screenAnchor: Vec2): void;
  /** Pans by a CSS-pixel screen delta. */
  panBy(screenDelta: Vec2): void;
  /** Centers and zooms to fit `documentRect` within `viewportSize` (minus `padding`). */
  fitToView(documentRect: Rect, viewportSize: Size, padding?: number): void;
  /** Records the CSS viewport size and device-pixel ratio (dpr clamped to {@link MAX_DPR}). */
  setViewportSize(width: number, height: number, dpr: number): void;
  /** Subscribes to any view change. Returns an unsubscribe function. */
  subscribe(listener: () => void): () => void;
}

/** Default fit-to-view padding in CSS pixels. */
const DEFAULT_FIT_PADDING = 48;

/** Creates a viewport, optionally seeded with an initial view state / size. */
export const createViewport = (initial?: Partial<ViewState>): Viewport => {
  let pan: Vec2 = { x: initial?.pan?.x ?? 0, y: initial?.pan?.y ?? 0 };
  let zoom = clampZoom(initial?.zoom ?? 1);
  let size: Size = { height: 0, width: 0 };
  let dpr = 1;

  const listeners = new Set<() => void>();

  const emit = (): void => {
    for (const listener of listeners) {
      listener();
    }
  };

  const cssMatrix = (): Mat2d => ({ a: zoom, b: 0, c: 0, d: zoom, e: pan.x, f: pan.y });

  const documentToScreen = (p: Vec2): Vec2 => ({ x: zoom * p.x + pan.x, y: zoom * p.y + pan.y });

  const screenToDocument = (p: Vec2): Vec2 => {
    const inverse = invert(cssMatrix());
    if (!inverse) {
      return { x: 0, y: 0 };
    }
    return applyToPoint(inverse, p);
  };

  const zoomAtPoint = (newZoom: number, screenAnchor: Vec2): void => {
    const next = calculateZoomAtPoint({ pan, zoom }, newZoom, screenAnchor, clampZoom);
    zoom = next.zoom;
    pan = next.pan;
    emit();
  };

  const wheelZoom = (deltaY: number, screenAnchor: Vec2): void => {
    const next = calculateWheelZoomAtPoint({ pan, zoom }, deltaY, screenAnchor, {
      constrainZoom: clampZoom,
      snapZoom,
    });
    zoom = next.zoom;
    pan = next.pan;
    emit();
  };

  const panBy = (screenDelta: Vec2): void => {
    const next = calculatePanBy({ pan, zoom }, screenDelta);
    pan = next.pan;
    emit();
  };

  const fitToView = (documentRect: Rect, viewportSize: Size, padding: number = DEFAULT_FIT_PADDING): void => {
    const availWidth = viewportSize.width - padding * 2;
    const availHeight = viewportSize.height - padding * 2;
    if (documentRect.width <= 0 || documentRect.height <= 0 || availWidth <= 0 || availHeight <= 0) {
      return;
    }
    const fitZoom = clampZoom(Math.min(availWidth / documentRect.width, availHeight / documentRect.height));
    const docCenter: Vec2 = {
      x: documentRect.x + documentRect.width / 2,
      y: documentRect.y + documentRect.height / 2,
    };
    zoom = fitZoom;
    pan = { x: viewportSize.width / 2 - fitZoom * docCenter.x, y: viewportSize.height / 2 - fitZoom * docCenter.y };
    emit();
  };

  const setViewportSize = (width: number, height: number, nextDpr: number): void => {
    const clampedDpr = Math.max(1, Math.min(MAX_DPR, nextDpr));
    if (size.width === width && size.height === height && dpr === clampedDpr) {
      return;
    }
    size = { height, width };
    dpr = clampedDpr;
    emit();
  };

  return {
    documentToScreen,
    fitToView,
    getDpr: () => dpr,
    getState: () => ({ pan: { x: pan.x, y: pan.y }, zoom }),
    getViewportSize: () => ({ height: size.height, width: size.width }),
    getZoom: () => zoom,
    panBy,
    screenToDocument,
    setViewportSize,
    subscribe: (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    viewMatrix: (matrixDpr) => ({
      a: zoom * matrixDpr,
      b: 0,
      c: 0,
      d: zoom * matrixDpr,
      e: pan.x * matrixDpr,
      f: pan.y * matrixDpr,
    }),
    wheelZoom,
    zoomAtPoint,
  };
};
