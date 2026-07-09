/**
 * Renders the interaction overlay — the second stacked canvas above the
 * composited document: bbox rectangle, an optional viewport-wide grid, and
 * the brush cursor ring. Everything is drawn in screen space (document points
 * projected through `view`) so strokes stay a constant pixel width regardless
 * of zoom. The canvas is an unbounded plane, so no document-bounds outline is
 * drawn.
 *
 * Marching ants and transform handles come later; the functions here are kept
 * small and composable so those can slot in. Every pixel operation flows
 * through the {@link RasterSurface} `ctx` seam. Zero React, zero import-time
 * side effects.
 */

import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint, getScale, invert } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';
import { drawMarchingAnts, type MarchingAntsRender } from '@workbench/canvas-engine/selection/marchingAnts';
import { BBOX_HANDLES, bboxHandlePoint } from '@workbench/canvas-engine/tools/bboxHitTest';
import { TRANSFORM_ROTATE_NUB_PX } from '@workbench/canvas-engine/transform/transformMath';

import type { RasterSurface } from './raster';

/** Minimum screen-space grid spacing (px) below which the grid is too dense to draw. */
export const MIN_GRID_SPACING_PX = 8;

const BBOX_COLOR = '#3b82f6';
const BBOX_DASH: readonly number[] = [4, 4];
/** The bbox-overlay dim fill (legacy `CanvasBboxToolModule` overlayRect parity). */
const BBOX_OVERLAY_FILL = 'hsl(220 12% 10% / 0.8)';
const GRID_COLOR = 'rgba(128, 128, 128, 0.25)';
const CURSOR_COLOR = '#ffffff';
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

/** Everything the overlay needs to draw a frame. */
export interface OverlayState {
  /** Document→screen transform. */
  view: Mat2d;
  /** Generation bounding box in document space. */
  bbox: Rect;
  /** Whether to draw the eight bbox resize handles (bbox tool active). */
  bboxHandles?: boolean;
  /** Whether to draw the passive bbox frame (default when absent: drawn). */
  showBbox?: boolean;
  /** Whether to dim everything outside the bbox (the legacy "bbox overlay" shade). */
  bboxOverlay?: boolean;
  /** Whether to draw the rule-of-thirds guides inside the bbox. */
  ruleOfThirds?: boolean;
  /** Whether to draw the grid. */
  showGrid?: boolean;
  /** Grid spacing in document units. */
  gridSize?: number;
  /** Brush cursor ring, or `null`/absent to hide it. */
  cursor?: OverlayCursor | null;
  /**
   * A layer's rendered-bounds outline (the move tool's selection marquee): the
   * four document-space corners, projected through `view` and stroked as a closed
   * polygon. Absent/`null` to hide it.
   */
  layerOutline?: readonly Vec2[] | null;
  /**
   * The active transform-tool frame: the layer's rotated bounds, its eight scale
   * handles, and the center/top-edge points for the rotation nub — all in
   * document space, projected through `view`. Absent/`null` when no transform
   * session is active.
   */
  transformFrame?: TransformFrameOverlay | null;
  /**
   * The in-progress lasso polygon (document-space points), drawn as a live dashed
   * outline while a lasso drag is underway. Absent/`null` when idle.
   */
  lassoPreview?: readonly Vec2[] | null;
  /**
   * The committed selection's marching ants: the outline paths (document space)
   * plus the animated dash phase. Absent/`null` when there is no selection.
   */
  marchingAnts?: MarchingAntsRender | null;
  /**
   * The in-progress shape-tool drag (document-space rect + kind), drawn as a
   * live outline while a shape is being created. Absent/`null` when idle.
   */
  shapePreview?: { rect: Rect; kind: 'rect' | 'ellipse' } | null;
  /**
   * The in-progress gradient-tool drag vector (document-space start/end),
   * drawn as a direction indicator. Absent/`null` when idle.
   */
  gradientPreview?: { start: Vec2; end: Vec2 } | null;
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
 * Draws grid lines across the entire viewport, in screen space, skipping when
 * too dense. The canvas is an unbounded plane, so the grid is not clipped to any
 * document rect: the visible screen rect is projected back into document space
 * (via the inverse view) and snapped out to whole grid cells so the lines tile
 * seamlessly while panning/zooming.
 */
const drawGrid = (ctx: Ctx, state: OverlayState, target: RasterSurface): void => {
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
    applyToPoint(inv, { x: target.width, y: 0 }),
    applyToPoint(inv, { x: 0, y: target.height }),
    applyToPoint(inv, { x: target.width, y: target.height }),
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
  ctx.strokeStyle = CURSOR_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(center.x, center.y, Math.max(0, radius), 0, Math.PI * 2);
  ctx.stroke();
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
 * Draws the transform frame (rotated bounds polygon, eight scale handles, and a
 * rotation nub past the top edge) in screen space. Handles/nub stay a constant
 * pixel size; the nub direction follows the rotated frame (top-edge → away from
 * center), so it reads correctly at any layer rotation.
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

  // Rotation nub: from the top-edge midpoint, outward (away from center), a
  // constant screen length; a small knob at the tip.
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

/** Draws the in-progress lasso polygon as a dashed screen-space outline. */
const drawLassoPreview = (ctx: Ctx, state: OverlayState): void => {
  const points = state.lassoPreview;
  if (!points || points.length < 2) {
    return;
  }
  ctx.save();
  ctx.strokeStyle = LASSO_PREVIEW_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...LASSO_PREVIEW_DASH]);
  ctx.beginPath();
  points.forEach((point, index) => {
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
  ctx.restore();
};

/** Draws the shape-tool drag preview (rect or ellipse outline) in screen space. */
const drawShapePreview = (ctx: Ctx, state: OverlayState): void => {
  const preview = state.shapePreview;
  if (!preview || preview.rect.width <= 0 || preview.rect.height <= 0) {
    return;
  }
  const screen = transformBounds(state.view, preview.rect);
  ctx.save();
  ctx.strokeStyle = LAYER_OUTLINE_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...BBOX_DASH]);
  ctx.beginPath();
  if (preview.kind === 'ellipse') {
    ctx.ellipse(
      screen.x + screen.width / 2,
      screen.y + screen.height / 2,
      Math.abs(screen.width) / 2,
      Math.abs(screen.height) / 2,
      0,
      0,
      Math.PI * 2
    );
  } else {
    ctx.rect(screen.x, screen.y, screen.width, screen.height);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
};

/** Draws the gradient-tool drag vector (a line with endpoint dots) in screen space. */
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
  for (const point of [start, end]) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
};

/**
 * Draws the bbox overlay shade: a translucent dark fill over the ENTIRE viewport
 * with the bbox region punched out (even-odd fill rule — the two nested rect paths
 * cancel inside the bbox), dimming everything outside the generation frame. Fill
 * color matches legacy `CanvasBboxToolModule`'s overlayRect.
 */
const drawBboxOverlayShade = (ctx: Ctx, state: OverlayState, target: RasterSurface): void => {
  if (!state.bboxOverlay) {
    return;
  }
  const bboxScreen = transformBounds(state.view, state.bbox);
  ctx.save();
  ctx.fillStyle = BBOX_OVERLAY_FILL;
  ctx.beginPath();
  ctx.rect(0, 0, target.width, target.height);
  ctx.rect(bboxScreen.x, bboxScreen.y, bboxScreen.width, bboxScreen.height);
  ctx.fill('evenodd');
  ctx.restore();
};

/**
 * Draws the rule-of-thirds guides: two vertical and two horizontal lines dividing
 * the bbox into thirds (screen space, solid, in the grid color). A composition aid
 * over the generation frame.
 */
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

/** Clears and redraws the entire overlay for the given `state`. */
export const renderOverlay = (target: RasterSurface, state: OverlayState): void => {
  const ctx = target.ctx;

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, target.width, target.height);

  // The document rect is retired as a visual boundary (unbounded plane), so no
  // document outline is drawn. The bbox (generation frame) is the primary anchor.

  // The bbox overlay shade first: it dims the composited document outside the
  // bbox, and every piece of overlay chrome (grid, guides, frame) draws over it.
  drawBboxOverlayShade(ctx, state, target);

  // Grid next (behind the bbox), spanning the whole viewport.
  if (state.showGrid) {
    drawGrid(ctx, state, target);
  }

  // Rule-of-thirds guides sit inside the bbox, behind its frame.
  drawRuleOfThirds(ctx, state);

  // Bbox (dashed, distinct color). Drawn as passive chrome unless the setting hides
  // it (`showBbox === false`); the handles below still render for the bbox tool so
  // the frame stays editable even when hidden.
  if (state.showBbox ?? true) {
    ctx.strokeStyle = BBOX_COLOR;
    ctx.setLineDash([...BBOX_DASH]);
    strokeRectScreen(ctx, transformBounds(state.view, state.bbox));
    ctx.setLineDash([]);
  }

  drawLayerOutline(ctx, state);
  drawBboxHandles(ctx, state);
  drawTransformFrame(ctx, state);
  if (state.marchingAnts) {
    drawMarchingAnts(ctx, state.view, state.marchingAnts);
  }
  drawLassoPreview(ctx, state);
  drawShapePreview(ctx, state);
  drawGradientPreview(ctx, state);
  drawCursor(ctx, state);

  ctx.restore();
};
