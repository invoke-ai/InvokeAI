/**
 * Interaction overlays render in CSS pixels through `view` under a device-pixel-ratio base transform, keeping
 * strokes and handles constant in CSS pixels at any zoom and DPR, where pointer hit testing measures them. The
 * unbounded plane has no document outline; all drawing uses {@link RasterSurface} contexts.
 */

import type { ParametricShapeKind } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, getScale, invert, multiply } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { buildParametricShapePath } from '@workbench/canvas-engine/render/rasterizers/shapeRasterizer';
import { drawMarchingAnts, type MarchingAntsRender } from '@workbench/canvas-engine/selection/marchingAnts';
import { BBOX_HANDLES, bboxHandlePoint } from '@workbench/canvas-engine/tools/bboxHitTest';
import { TRANSFORM_ROTATE_NUB_PX } from '@workbench/canvas-engine/transform/transformMath';

import type { RgbaSample } from './colorSample';
import type { RasterSurface } from './raster';

/** Minimum screen-space grid spacing (px) below which the grid is too dense to draw. */
export const MIN_GRID_SPACING_PX = 8;

const BBOX_COLOR = '#3b82f6';
const BBOX_DASH: readonly number[] = [4, 4];
/** The bbox-overlay dim fill when no theme surround color has been fed in. */
const BBOX_OVERLAY_FILL = 'hsl(220 12% 10%)';
/** The shade is the surround color at this opacity: the dimmed area reads as "outside the document". */
const BBOX_OVERLAY_ALPHA = 0.8;
const GRID_COLOR = 'rgba(128, 128, 128, 0.25)';
const CURSOR_DARK = '#000000';
const CURSOR_LIGHT = '#ffffff';
const CURSOR_OUTER_WIDTH_PX = 3;
const CURSOR_INNER_WIDTH_PX = 1;
/** Side length (screen px) of a drawn bbox resize handle. */
const BBOX_HANDLE_DRAW_PX = 8;
const BBOX_HANDLE_FILL = '#ffffff';
/** Solid accent used for the selected-layer bounds outline while the move tool is active. */
const LAYER_OUTLINE_COLOR = '#38bdf8';
/** Accent for the transform frame (outline, handles, rotation nub). */
const TRANSFORM_COLOR = '#38bdf8';
const TRANSFORM_HANDLE_FILL = '#ffffff';
/** Side length (screen px) of a drawn transform scale handle. */
const TRANSFORM_HANDLE_DRAW_PX = 8;
/** Screen-px radius of the rotation-indicator knob at the nub's tip. */
const TRANSFORM_ROTATE_KNOB_PX = 3.5;

/** The brush cursor ring: center in document space, radius in document units. */
export interface OverlayCursor {
  point: Vec2;
  radiusDoc: number;
}

/** The color picker's loupe diameter in CSS pixels. */
const COLOR_LOUPE_DIAMETER_PX = 120;
/** Each magnified pixel spans at least this many CSS pixels, and twice the canvas zoom. */
const COLOR_LOUPE_MIN_CELL_PX = 8;
const COLOR_LOUPE_RING = '#d4d4d4';
const COLOR_LOUPE_LABEL_FONT = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
const COLOR_LOUPE_LABEL_FILL = 'rgba(0, 0, 0, 0.72)';

/** Document pixels across the loupe at `zoom`: odd, so the sampled one sits at the center, and at least three. */
export const colorLoupePixels = (zoom: number): number => {
  const cell = Math.max(COLOR_LOUPE_MIN_CELL_PX, zoom * 2);
  const count = Math.floor(COLOR_LOUPE_DIAMETER_PX / cell);
  return Math.max(3, count % 2 === 0 ? count - 1 : count);
};

/** The color picker's magnified view of the pixels under the pointer. */
export interface ColorLoupeOverlay {
  /** The pointer in document space; the loupe is centered on it. */
  point: Vec2;
  /** A square of document pixels around the sampled one, which is the center pixel. */
  pixels: RasterSurface;
  /** The sampled color, or null over empty canvas. */
  color: RgbaSample | null;
  /** The canvas checkerboard tile, shown where the canvas is empty. */
  checker: RasterSurface;
}

/** A live parametric-shape drag outline in document space (shape and marquee tools). */
export interface RectShapePreview {
  rect: Rect;
  kind: ParametricShapeKind;
}

/** Everything the overlay needs to draw a frame. */
export interface OverlayState {
  /** Document→CSS-pixel transform. */
  view: Mat2d;
  /** Backing-store pixels per CSS pixel. */
  dpr: number;
  /** The viewport in CSS pixels. */
  viewportSize: { readonly width: number; readonly height: number };
  /** Generation bounding box in document space. */
  bbox: Rect;
  /** Whether to draw the eight bbox resize handles (bbox tool active). */
  bboxHandles?: boolean;
  /** Whether to draw the passive bbox frame (default when absent: drawn). */
  showBbox?: boolean;
  /** Whether to dim everything outside the bbox (the legacy "bbox overlay" shade). */
  bboxOverlay?: boolean;
  /** Opaque CSS color for that shade — the theme's canvas surround, resolved by the widget. */
  bboxOverlayColor?: string;
  /** Whether to draw the rule-of-thirds guides inside the bbox. */
  ruleOfThirds?: boolean;
  /** Whether to draw the grid. */
  showGrid?: boolean;
  /** Grid spacing in document units. */
  gridSize?: number;
  /** Brush cursor ring, or `null`/absent to hide it. */
  cursor?: OverlayCursor | null;
  /** The color picker's loupe, or `null`/absent to hide it. */
  colorLoupe?: ColorLoupeOverlay | null;
  /** Move outline uses four document-space content corners projected through view; null/absence hides it. */
  layerOutline?: readonly Vec2[] | null;
  /**
   * Document-space rotated transform bounds, eight handles and rotation-nub anchors; null/absence means no
   * session.
   */
  transformFrame?: TransformFrameOverlay | null;
  /** Document-space freehand/polygon lasso preview with vertices and close cue; null/absence means idle. */
  lassoPreview?:
    | { kind: 'freehand'; points: readonly Vec2[] }
    | {
        kind: 'polygon';
        points: readonly Vec2[];
        cursor: Vec2 | null;
        closeRadiusPx: number | null;
        closeArmed: boolean;
      }
    | null;
  /** Committed document-space selection paths and animated dash phase; absent without selection. */
  marchingAnts?: MarchingAntsRender | null;
  /** Document-space shape creation preview; null/absence means idle. */
  shapePreview?: RectShapePreview | null;
  /** Document-space marquee drag preview; committed selections instead use ants. */
  marqueePreview?: RectShapePreview | null;
  /** Gradient drag start/end defines the linear vector or radial center/radius; absent when idle. */
  gradientPreview?: { kind: 'linear' | 'radial'; start: Vec2; end: Vec2 } | null;
  /** Dedicated Select Object mask preview, already colorized by the engine. */
  samPreview?: {
    surface: RasterSurface;
    rect: Rect;
    opacity: number;
    /** True-edge outline for marching ants, document space; `null` skips the ants. */
    outline: Path2D | null;
    phase: number;
  } | null;
  /** Select Object visual prompt geometry in document space. */
  samInput?: { includePoints: readonly Vec2[]; excludePoints: readonly Vec2[]; bbox: Rect | null } | null;
}

/** Document-space geometry the overlay draws for an active transform session. */
export interface TransformFrameOverlay {
  /** The four rotated-rect corners (closed polygon). */
  corners: readonly Vec2[];
  /** The eight scale-handle positions. */
  handles: readonly Vec2[];
  /** The layer center (rotation nub direction reference). */
  center: Vec2;
  /** The top edge midpoint (root of the rotation nub). */
  rotationAnchor: Vec2;
}

type Ctx = RasterSurface['ctx'];

const strokeRectScreen = (ctx: Ctx, screenRect: Rect): void => {
  ctx.strokeRect(screenRect.x, screenRect.y, screenRect.width, screenRect.height);
};

/**
 * Project viewport bounds into document space and snap outward to grid cells for seamless pan/zoom. Skip overly
 * dense grids; never clip to document bounds.
 */
const drawGrid = (ctx: Ctx, state: OverlayState): void => {
  const gridSize = state.gridSize ?? 0;
  if (gridSize <= 0) {
    return;
  }
  const scale = getScale(state.view);
  if (gridSize * scale < MIN_GRID_SPACING_PX) {
    return;
  }

  const { view } = state;
  const inv = invert(view);
  if (!inv) {
    return;
  }
  // Document-space bounds of the viewport's four screen corners.
  const corners = [
    applyToPoint(inv, { x: 0, y: 0 }),
    applyToPoint(inv, { x: state.viewportSize.width, y: 0 }),
    applyToPoint(inv, { x: 0, y: state.viewportSize.height }),
    applyToPoint(inv, { x: state.viewportSize.width, y: state.viewportSize.height }),
  ];
  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
  // Snap outward to whole grid cells so lines are stable across pan/zoom.
  const left = Math.floor(Math.min(...xs) / gridSize) * gridSize;
  const top = Math.floor(Math.min(...ys) / gridSize) * gridSize;
  const right = Math.ceil(Math.max(...xs) / gridSize) * gridSize;
  const bottom = Math.ceil(Math.max(...ys) / gridSize) * gridSize;

  ctx.save();
  ctx.strokeStyle = GRID_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  for (let x = left; x <= right; x += gridSize) {
    const a = applyToPoint(view, { x, y: top });
    const b = applyToPoint(view, { x, y: bottom });
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  for (let y = top; y <= bottom; y += gridSize) {
    const a = applyToPoint(view, { x: left, y });
    const b = applyToPoint(view, { x: right, y });
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
  ctx.restore();
};

/** Draws the brush cursor ring at the pointer, radius scaled from document units. */
const drawCursor = (ctx: Ctx, state: OverlayState): void => {
  const cursor = state.cursor;
  if (!cursor) {
    return;
  }
  const center = applyToPoint(state.view, cursor.point);
  const radius = cursor.radiusDoc * getScale(state.view);
  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, Math.max(0, radius), 0, Math.PI * 2);
  // A dark outer ring plus a light inner ring stays legible over either end of
  // the value range without sampling pixels or relying on blend-mode support.
  ctx.strokeStyle = CURSOR_DARK;
  ctx.lineWidth = CURSOR_OUTER_WIDTH_PX;
  ctx.stroke();
  ctx.strokeStyle = CURSOR_LIGHT;
  ctx.lineWidth = CURSOR_INNER_WIDTH_PX;
  ctx.stroke();
  ctx.restore();
};

/** Strokes the current path dark-then-light so it reads over any pixels. */
const strokeOutlined = (ctx: Ctx, light: string, lightWidth: number): void => {
  ctx.strokeStyle = CURSOR_DARK;
  ctx.lineWidth = lightWidth + 2;
  ctx.stroke();
  ctx.strokeStyle = light;
  ctx.lineWidth = lightWidth;
  ctx.stroke();
};

/**
 * Draws the picker's loupe: the pixels around the pointer magnified without smoothing inside a ring, the sampled
 * center pixel boxed as the target (the picker hides the system cursor), and its RGB values below.
 */
const drawColorLoupe = (ctx: Ctx, state: OverlayState): void => {
  const loupe = state.colorLoupe;
  if (!loupe) {
    return;
  }
  const point = applyToPoint(state.view, loupe.point);
  // Whole device pixels keep the magnified cells crisp.
  const center = { x: Math.round(point.x * state.dpr) / state.dpr, y: Math.round(point.y * state.dpr) / state.dpr };
  const radius = COLOR_LOUPE_DIAMETER_PX / 2;
  const cell = COLOR_LOUPE_DIAMETER_PX / loupe.pixels.width;
  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.clip();
  const checker = ctx.createPattern(loupe.checker.canvas, 'repeat');
  if (checker) {
    ctx.fillStyle = checker;
    ctx.fillRect(center.x - radius, center.y - radius, radius * 2, radius * 2);
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(loupe.pixels.canvas, center.x - radius, center.y - radius, radius * 2, radius * 2);
  ctx.restore();

  ctx.save();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.rect(center.x - cell / 2, center.y - cell / 2, cell, cell);
  strokeOutlined(ctx, CURSOR_LIGHT, 1);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  strokeOutlined(ctx, COLOR_LOUPE_RING, 2);

  if (loupe.color) {
    const label = `${loupe.color.r} ${loupe.color.g} ${loupe.color.b}`;
    ctx.font = COLOR_LOUPE_LABEL_FONT;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const width = ctx.measureText(label).width + 12;
    const height = 18;
    const labelY = center.y + cell / 2 + 12 + height / 2;
    const end = height / 2;
    ctx.beginPath();
    ctx.arc(center.x - width / 2 + end, labelY, end, Math.PI / 2, (Math.PI * 3) / 2);
    ctx.arc(center.x + width / 2 - end, labelY, end, -Math.PI / 2, Math.PI / 2);
    ctx.closePath();
    ctx.fillStyle = COLOR_LOUPE_LABEL_FILL;
    ctx.fill();
    ctx.fillStyle = CURSOR_LIGHT;
    ctx.fillText(label, center.x, labelY);
  }
  ctx.restore();
};

/** Draws a layer's rendered-bounds outline as a closed polygon in screen space. */
const drawLayerOutline = (ctx: Ctx, state: OverlayState): void => {
  const corners = state.layerOutline;
  if (!corners || corners.length < 2) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = LAYER_OUTLINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  corners.forEach((corner, index) => {
    const p = applyToPoint(state.view, corner);
    if (index === 0) {
      ctx.moveTo(p.x, p.y);
    } else {
      ctx.lineTo(p.x, p.y);
    }
  });
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
};

/**
 * Screen-space transform bounds and fixed-size handles. Rotation nub extends from the top edge away from center,
 * following layer rotation.
 */
const drawTransformFrame = (ctx: Ctx, state: OverlayState): void => {
  const frame = state.transformFrame;
  if (!frame || frame.corners.length < 3) {
    return;
  }
  const { view } = state;
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = TRANSFORM_COLOR;
  ctx.lineWidth = 1;

  // Bounds polygon.
  ctx.beginPath();
  frame.corners.forEach((corner, index) => {
    const p = applyToPoint(view, corner);
    if (index === 0) {
      ctx.moveTo(p.x, p.y);
    } else {
      ctx.lineTo(p.x, p.y);
    }
  });
  ctx.closePath();
  ctx.stroke();

  const anchor = applyToPoint(view, frame.rotationAnchor);
  const center = applyToPoint(view, frame.center);
  const dx = anchor.x - center.x;
  const dy = anchor.y - center.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = anchor.x + (dx / len) * TRANSFORM_ROTATE_NUB_PX;
  const ny = anchor.y + (dy / len) * TRANSFORM_ROTATE_NUB_PX;
  ctx.beginPath();
  ctx.moveTo(anchor.x, anchor.y);
  ctx.lineTo(nx, ny);
  ctx.stroke();
  ctx.beginPath();
  ctx.fillStyle = TRANSFORM_HANDLE_FILL;
  ctx.arc(nx, ny, TRANSFORM_ROTATE_KNOB_PX, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Scale handles.
  const half = TRANSFORM_HANDLE_DRAW_PX / 2;
  ctx.fillStyle = TRANSFORM_HANDLE_FILL;
  for (const handle of frame.handles) {
    const p = applyToPoint(view, handle);
    ctx.fillRect(p.x - half, p.y - half, TRANSFORM_HANDLE_DRAW_PX, TRANSFORM_HANDLE_DRAW_PX);
    ctx.strokeRect(p.x - half, p.y - half, TRANSFORM_HANDLE_DRAW_PX, TRANSFORM_HANDLE_DRAW_PX);
  }
  ctx.restore();
};

const LASSO_PREVIEW_COLOR = '#38bdf8';
const LASSO_PREVIEW_DASH: readonly number[] = [4, 4];

/** Screen-space half-size of a placed polygon vertex knob. */
const LASSO_VERTEX_HALF_PX = 2;

/** Dashed lasso preview; polygons add vertices and a first-vertex close-radius cue filled on hover. */
const drawLassoPreview = (ctx: Ctx, state: OverlayState): void => {
  const preview = state.lassoPreview;
  if (!preview) {
    return;
  }
  const outline = preview.kind === 'polygon' && preview.cursor ? [...preview.points, preview.cursor] : preview.points;
  if (outline.length < 2) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = LASSO_PREVIEW_COLOR;
  ctx.fillStyle = LASSO_PREVIEW_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...LASSO_PREVIEW_DASH]);
  ctx.beginPath();
  outline.forEach((point, index) => {
    const p = applyToPoint(state.view, point);
    if (index === 0) {
      ctx.moveTo(p.x, p.y);
    } else {
      ctx.lineTo(p.x, p.y);
    }
  });
  // Close the loop back to the start so the enclosed region reads clearly.
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  if (preview.kind === 'polygon') {
    preview.points.forEach((point, index) => {
      const p = applyToPoint(state.view, point);
      if (index === 0 && preview.closeRadiusPx !== null) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, preview.closeRadiusPx, 0, Math.PI * 2);
        if (preview.closeArmed) {
          ctx.fill();
        }
        ctx.stroke();
      } else {
        ctx.fillRect(
          p.x - LASSO_VERTEX_HALF_PX,
          p.y - LASSO_VERTEX_HALF_PX,
          LASSO_VERTEX_HALF_PX * 2,
          LASSO_VERTEX_HALF_PX * 2
        );
      }
    });
  }
  ctx.restore();
};

const drawRectShapePreview = (ctx: Ctx, state: OverlayState, preview: RectShapePreview | null | undefined): void => {
  if (!preview || preview.rect.width <= 0 || preview.rect.height <= 0) {
    return;
  }
  const screen = transformBounds(state.view, preview.rect);
  ctx.save();
  ctx.strokeStyle = LAYER_OUTLINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...BBOX_DASH]);
  // The same path the rasterizer commits, so the outline is the shape it makes.
  buildParametricShapePath(ctx, preview.kind, screen.x, screen.y, Math.abs(screen.width), Math.abs(screen.height), 0);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
};

const drawGradientPreview = (ctx: Ctx, state: OverlayState): void => {
  const preview = state.gradientPreview;
  if (!preview) {
    return;
  }
  const start = applyToPoint(state.view, preview.start);
  const end = applyToPoint(state.view, preview.end);
  ctx.save();
  ctx.strokeStyle = LAYER_OUTLINE_COLOR;
  ctx.fillStyle = LAYER_OUTLINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
  if (preview.kind === 'radial') {
    ctx.setLineDash([...BBOX_DASH]);
    ctx.beginPath();
    ctx.arc(start.x, start.y, Math.hypot(end.x - start.x, end.y - start.y), 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const point of [start, end]) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
};

const SAM_BBOX_COLOR = '#f59e0b';
const SAM_INCLUDE_COLOR = '#22c55e';
const SAM_EXCLUDE_COLOR = '#ef4444';
const SAM_POINT_RADIUS_PX = 5;
const SAM_HANDLE_DRAW_PX = 8;

const drawSamPreview = (ctx: Ctx, state: OverlayState, base: Mat2d): void => {
  const preview = state.samPreview;
  if (!preview) {
    return;
  }
  const placed = multiply(base, state.view);
  ctx.save();
  ctx.setTransform(placed.a, placed.b, placed.c, placed.d, placed.e, placed.f);
  ctx.globalAlpha = preview.opacity;
  ctx.drawImage(preview.surface.canvas, preview.rect.x, preview.rect.y, preview.rect.width, preview.rect.height);
  ctx.restore();
  if (preview.outline) {
    drawMarchingAnts(ctx, state.view, { matrix: null, paths: [preview.outline], phase: preview.phase }, base);
  }
};

const drawSamGeometry = (ctx: Ctx, state: OverlayState): void => {
  const input = state.samInput;
  if (!input) {
    return;
  }
  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = SAM_BBOX_COLOR;
  ctx.setLineDash([...BBOX_DASH]);
  if (input.bbox) {
    const screenRect = transformBounds(state.view, input.bbox);
    ctx.strokeRect(screenRect.x, screenRect.y, screenRect.width, screenRect.height);
    ctx.setLineDash([]);
    ctx.fillStyle = BBOX_HANDLE_FILL;
    const half = SAM_HANDLE_DRAW_PX / 2;
    for (const handle of BBOX_HANDLES) {
      const center = bboxHandlePoint(screenRect, handle);
      ctx.fillRect(center.x - half, center.y - half, SAM_HANDLE_DRAW_PX, SAM_HANDLE_DRAW_PX);
      ctx.strokeRect(center.x - half, center.y - half, SAM_HANDLE_DRAW_PX, SAM_HANDLE_DRAW_PX);
    }
  }
  ctx.setLineDash([]);
  for (const [points, color] of [
    [input.includePoints, SAM_INCLUDE_COLOR],
    [input.excludePoints, SAM_EXCLUDE_COLOR],
  ] as const) {
    ctx.fillStyle = color;
    ctx.strokeStyle = '#ffffff';
    for (const point of points) {
      const screen = applyToPoint(state.view, point);
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, SAM_POINT_RADIUS_PX, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
};

/**
 * Even-odd viewport/bbox fill shades outside the generation frame; globalAlpha controls opacity independently of
 * theme color.
 */
const drawBboxOverlayShade = (ctx: Ctx, target: RasterSurface, state: OverlayState): void => {
  if (!state.bboxOverlay) {
    return;
  }
  const bboxScreen = transformBounds(state.view, state.bbox);
  ctx.save();
  ctx.globalAlpha = BBOX_OVERLAY_ALPHA;
  ctx.fillStyle = state.bboxOverlayColor ?? BBOX_OVERLAY_FILL;
  ctx.beginPath();
  // The backing store rounds its device size, so cover it whole rather than the CSS viewport.
  ctx.rect(0, 0, target.width / state.dpr, target.height / state.dpr);
  ctx.rect(bboxScreen.x, bboxScreen.y, bboxScreen.width, bboxScreen.height);
  ctx.fill('evenodd');
  ctx.restore();
};

const drawRuleOfThirds = (ctx: Ctx, state: OverlayState): void => {
  if (!state.ruleOfThirds) {
    return;
  }
  const r = transformBounds(state.view, state.bbox);
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = GRID_COLOR;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i <= 2; i++) {
    const x = r.x + (r.width * i) / 3;
    const y = r.y + (r.height * i) / 3;
    ctx.moveTo(x, r.y);
    ctx.lineTo(x, r.y + r.height);
    ctx.moveTo(r.x, y);
    ctx.lineTo(r.x + r.width, y);
  }
  ctx.stroke();
  ctx.restore();
};

/** Draws the eight bbox resize handles as small squares at the frame's corners/edges (screen space). */
const drawBboxHandles = (ctx: Ctx, state: OverlayState): void => {
  if (!state.bboxHandles) {
    return;
  }
  const screenRect = transformBounds(state.view, state.bbox);
  const half = BBOX_HANDLE_DRAW_PX / 2;
  ctx.save();
  ctx.setLineDash([]);
  ctx.fillStyle = BBOX_HANDLE_FILL;
  ctx.strokeStyle = BBOX_COLOR;
  ctx.lineWidth = 1;
  for (const handle of BBOX_HANDLES) {
    const center = bboxHandlePoint(screenRect, handle);
    ctx.fillRect(center.x - half, center.y - half, BBOX_HANDLE_DRAW_PX, BBOX_HANDLE_DRAW_PX);
    ctx.strokeRect(center.x - half, center.y - half, BBOX_HANDLE_DRAW_PX, BBOX_HANDLE_DRAW_PX);
  }
  ctx.restore();
};

/** Clears the backing store, then redraws the overlay in CSS pixels under the state's device-pixel ratio. */
export const renderOverlay = (target: RasterSurface, state: OverlayState): void => {
  const ctx = target.ctx;
  const base: Mat2d = { a: state.dpr, b: 0, c: 0, d: state.dpr, e: 0, f: 0 };

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, target.width, target.height);
  ctx.setTransform(base.a, base.b, base.c, base.d, base.e, base.f);

  // Shade outside bbox first, then draw overlay chrome above it. The unbounded plane has no document outline.
  drawBboxOverlayShade(ctx, target, state);

  // Grid next (behind the bbox), spanning the whole viewport.
  if (state.showGrid) {
    drawGrid(ctx, state);
  }

  // Rule-of-thirds guides sit inside the bbox, behind its frame.
  drawRuleOfThirds(ctx, state);

  // Passive bbox visibility is optional; active-tool handles remain editable even when hidden.
  if (state.showBbox ?? true) {
    ctx.strokeStyle = BBOX_COLOR;
    ctx.setLineDash([...BBOX_DASH]);
    strokeRectScreen(ctx, transformBounds(state.view, state.bbox));
    ctx.setLineDash([]);
  }

  drawSamPreview(ctx, state, base);
  drawSamGeometry(ctx, state);
  drawLayerOutline(ctx, state);
  drawBboxHandles(ctx, state);
  drawTransformFrame(ctx, state);
  if (state.marchingAnts) {
    drawMarchingAnts(ctx, state.view, state.marchingAnts, base);
  }
  drawLassoPreview(ctx, state);
  drawRectShapePreview(ctx, state, state.marqueePreview);
  drawRectShapePreview(ctx, state, state.shapePreview);
  drawGradientPreview(ctx, state);
  drawCursor(ctx, state);
  drawColorLoupe(ctx, state);

  ctx.restore();
};
