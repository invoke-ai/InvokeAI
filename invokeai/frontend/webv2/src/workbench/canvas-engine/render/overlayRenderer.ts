/** Overlays draw in CSS pixels under a DPR base transform, so chrome stays constant at any zoom, matching hit tests. */

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
import type { VectorEditOverlay } from './vectorEditOverlay';
import type { VectorGuideLayer, VectorShapePreview } from './vectorGuides';

import { drawPolylineVertices, SHAPE_PREVIEW_COLOR, SHAPE_PREVIEW_DASH } from './rasterizers/shapeRasterizer';

export const MIN_GRID_SPACING_PX = 8;

const BBOX_COLOR = '#3b82f6';
const BBOX_DASH: readonly number[] = [4, 4];
/** Accent hairline cased navy outside and white inside, so some band of every dash keeps about 3.8:1 contrast. */
const BBOX_CASING_OUTSIDE = '#172554';
const BBOX_CASING_INSIDE = '#ffffff';
/** CSS px; each edge fits a whole odd number of dashes and gaps, dashes at both corners. */
const BBOX_FRAME_DASH_PX = 4;
/** Amber, so a frame moved off them still reads as a different thing. */
const STAGED_REGION_COLOR = '#f59e0b';
const STAGED_REGION_DASH: readonly number[] = [6, 4];
const STAGED_REGION_ALPHA = 0.8;
const STAGED_REGION_WIDTH_PX = 1.5;
const STAGED_REGION_SELECTED_WIDTH_PX = 2.5;
const STAGED_REGION_HALO = 'rgba(0, 0, 0, 0.55)';
const BBOX_OVERLAY_FILL = 'hsl(220 12% 10%)';
const BBOX_OVERLAY_ALPHA = 0.8;
const GRID_COLOR = 'rgba(128, 128, 128, 0.25)';
const CURSOR_DARK = '#000000';
const CURSOR_LIGHT = '#ffffff';
const CURSOR_OUTER_WIDTH_PX = 3;
const CURSOR_INNER_WIDTH_PX = 1;
const BBOX_HANDLE_FACE_PX = 7;
const BBOX_HANDLE_FILL = '#ffffff';
const LAYER_OUTLINE_COLOR = '#38bdf8';
const TRANSFORM_COLOR = '#38bdf8';
const TRANSFORM_HANDLE_FILL = '#ffffff';
const TRANSFORM_HANDLE_DRAW_PX = 8;
const TRANSFORM_ROTATE_KNOB_PX = 3.5;

/** Center in document space, radius in document units. */
export interface OverlayCursor {
  point: Vec2;
  radiusDoc: number;
}

const COLOR_LOUPE_DIAMETER_PX = 120;
/** Each magnified pixel spans at least this many CSS pixels, and twice the canvas zoom. */
const COLOR_LOUPE_MIN_CELL_PX = 8;
const COLOR_LOUPE_RING = '#d4d4d4';
const COLOR_LOUPE_LABEL_FONT = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
const COLOR_LOUPE_LABEL_FILL = 'rgba(0, 0, 0, 0.72)';

/** Odd, so the sampled pixel sits at the center, and at least three. */
export const colorLoupePixels = (zoom: number): number => {
  const cell = Math.max(COLOR_LOUPE_MIN_CELL_PX, zoom * 2);
  const count = Math.floor(COLOR_LOUPE_DIAMETER_PX / cell);
  return Math.max(3, count % 2 === 0 ? count - 1 : count);
};

export interface ColorLoupeOverlay {
  point: Vec2;
  pixels: RasterSurface;
  /** Null over empty canvas. */
  color: RgbaSample | null;
  checker: RasterSurface;
}

export interface RectShapePreview {
  rect: Rect;
  kind: ParametricShapeKind;
}

export interface OverlayState {
  vectorEdit?: VectorEditOverlay | null;
  vectorGuides?: readonly VectorGuideLayer[];
  vectorShapePreview?: VectorShapePreview | null;
  /** Document→CSS-pixel transform. */
  view: Mat2d;
  dpr: number;
  viewportSize: { readonly width: number; readonly height: number };
  bbox: Rect;
  stagedRegions?: readonly { readonly rect: Rect; readonly selected: boolean }[];
  bboxHandles?: boolean;
  /** Absent means drawn. */
  showBbox?: boolean;
  bboxOverlay?: boolean;
  /** The theme's canvas surround, resolved by the widget. */
  bboxOverlayColor?: string;
  ruleOfThirds?: boolean;
  showGrid?: boolean;
  gridSize?: number;
  cursor?: OverlayCursor | null;
  colorLoupe?: ColorLoupeOverlay | null;
  /** Four document-space content corners. */
  layerOutline?: readonly Vec2[] | null;
  transformFrame?: TransformFrameOverlay | null;
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
  marchingAnts?: MarchingAntsRender | null;
  shapePreview?: RectShapePreview | null;
  marqueePreview?: RectShapePreview | null;
  gradientPreview?: { kind: 'linear' | 'radial'; start: Vec2; end: Vec2 } | null;
  /** Already colorized by the engine. */
  samPreview?: {
    surface: RasterSurface;
    rect: Rect;
    opacity: number;
    /** `null` skips the ants. */
    outline: Path2D | null;
    phase: number;
  } | null;
  samInput?: { includePoints: readonly Vec2[]; excludePoints: readonly Vec2[]; bbox: Rect | null } | null;
}

export interface TransformFrameOverlay {
  corners: readonly Vec2[];
  handles: readonly Vec2[];
  center: Vec2;
  rotationAnchor: Vec2;
}

type Ctx = RasterSurface['ctx'];

const strokeRectScreen = (ctx: Ctx, screenRect: Rect): void => {
  ctx.strokeRect(screenRect.x, screenRect.y, screenRect.width, screenRect.height);
};

/** Snaps outward to grid cells for seamless pan and zoom; never clips to document bounds. */
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
  const corners = [
    applyToPoint(inv, { x: 0, y: 0 }),
    applyToPoint(inv, { x: state.viewportSize.width, y: 0 }),
    applyToPoint(inv, { x: 0, y: state.viewportSize.height }),
    applyToPoint(inv, { x: state.viewportSize.width, y: state.viewportSize.height }),
  ];
  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
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
  // Dark outer plus light inner ring stays legible over any pixels without sampling or blend modes.
  ctx.strokeStyle = CURSOR_DARK;
  ctx.lineWidth = CURSOR_OUTER_WIDTH_PX;
  ctx.stroke();
  ctx.strokeStyle = CURSOR_LIGHT;
  ctx.lineWidth = CURSOR_INNER_WIDTH_PX;
  ctx.stroke();
  ctx.restore();
};

const strokeOutlined = (ctx: Ctx, light: string, lightWidth: number): void => {
  ctx.strokeStyle = CURSOR_DARK;
  ctx.lineWidth = lightWidth + 2;
  ctx.stroke();
  ctx.strokeStyle = light;
  ctx.lineWidth = lightWidth;
  ctx.stroke();
};

/** The picker hides the system cursor, so the center pixel is boxed as the target. */
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

  const half = TRANSFORM_HANDLE_DRAW_PX / 2;
  ctx.fillStyle = TRANSFORM_HANDLE_FILL;
  for (const handle of frame.handles) {
    const p = applyToPoint(view, handle);
    ctx.fillRect(p.x - half, p.y - half, TRANSFORM_HANDLE_DRAW_PX, TRANSFORM_HANDLE_DRAW_PX);
    ctx.strokeRect(p.x - half, p.y - half, TRANSFORM_HANDLE_DRAW_PX, TRANSFORM_HANDLE_DRAW_PX);
  }
  ctx.restore();
};

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
  ctx.strokeStyle = SHAPE_PREVIEW_COLOR;
  ctx.fillStyle = SHAPE_PREVIEW_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...SHAPE_PREVIEW_DASH]);
  ctx.beginPath();
  outline.forEach((point, index) => {
    const p = applyToPoint(state.view, point);
    if (index === 0) {
      ctx.moveTo(p.x, p.y);
    } else {
      ctx.lineTo(p.x, p.y);
    }
  });
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  if (preview.kind === 'polygon') {
    drawPolylineVertices(ctx, state.view, preview.points, preview.closeRadiusPx, preview.closeArmed);
  }
  ctx.restore();
};

const drawRectShapePreview = (ctx: Ctx, state: OverlayState, preview: RectShapePreview | null | undefined): void => {
  if (!preview || preview.rect.width <= 0 || preview.rect.height <= 0) {
    return;
  }
  const screen = transformBounds(state.view, preview.rect);
  ctx.save();
  ctx.strokeStyle = SHAPE_PREVIEW_COLOR;
  ctx.lineWidth = 1;
  ctx.setLineDash([...SHAPE_PREVIEW_DASH]);
  // The same path the rasterizer commits, so the outline is the shape it makes.
  buildParametricShapePath(ctx, preview.kind, screen.x, screen.y, Math.abs(screen.width), Math.abs(screen.height));
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

/** `PreviewStateController` already lists each rect once with the selected one last. */
const drawStagedRegions = (ctx: Ctx, state: OverlayState): void => {
  const regions = state.stagedRegions;
  if (!regions || regions.length === 0) {
    return;
  }
  ctx.save();
  for (const { rect, selected } of regions) {
    const screenRect = transformBounds(state.view, rect);
    const width = selected ? STAGED_REGION_SELECTED_WIDTH_PX : STAGED_REGION_WIDTH_PX;
    ctx.globalAlpha = selected ? 1 : STAGED_REGION_ALPHA;
    ctx.setLineDash([]);
    ctx.strokeStyle = STAGED_REGION_HALO;
    ctx.lineWidth = width + 2;
    strokeRectScreen(ctx, screenRect);
    ctx.setLineDash([...STAGED_REGION_DASH]);
    ctx.strokeStyle = STAGED_REGION_COLOR;
    ctx.lineWidth = width;
    strokeRectScreen(ctx, screenRect);
  }
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

/** Core lines sit on whole device pixels, so a 1px line never smears into two half-strength pixels. */
interface BboxFrameGeometry {
  rect: Rect;
  /** About one CSS pixel, never thinner than one device pixel. */
  hairline: number;
  /** Dashes beyond it are skipped, so a deep zoom traces only what shows. */
  width: number;
  height: number;
}

/** Ties go toward `outward`, keeping a whole-pixel frame's core just outside the area it marks. */
const snapCentre = (device: number, odd: boolean, outward: -1 | 1): number => {
  if (odd) {
    return outward < 0 ? Math.ceil(device) - 0.5 : Math.floor(device) + 0.5;
  }
  return outward < 0 ? Math.ceil(device - 0.5) : Math.floor(device + 0.5);
};

const bboxFrameGeometry = (target: RasterSurface, state: OverlayState): BboxFrameGeometry => {
  const screen = transformBounds(state.view, state.bbox);
  const { dpr } = state;
  const hairline = Math.max(1, Math.round(dpr));
  const odd = hairline % 2 === 1;
  const left = snapCentre(screen.x * dpr, odd, -1);
  const top = snapCentre(screen.y * dpr, odd, -1);
  const right = snapCentre((screen.x + screen.width) * dpr, odd, 1);
  const bottom = snapCentre((screen.y + screen.height) * dpr, odd, 1);
  return {
    height: target.height,
    hairline,
    rect: { height: bottom - top, width: right - left, x: left, y: top },
    width: target.width,
  };
};

/** Edges run clockwise from the top-left: 0 top, 1 right, 2 bottom, 3 left; each starts at its own corner. */
const frameEdgeLength = (rect: Rect, edge: number): number => (edge % 2 === 0 ? rect.width : rect.height);

/** An odd count, so the edge starts and ends on a dash. */
const frameEdgeSegments = (rect: Rect, edge: number, dash: number): number =>
  Math.max(1, Math.round((frameEdgeLength(rect, edge) / dash + 1) / 2)) * 2 - 1;

const frameBoundary = (rect: Rect, edge: number, k: number, segments: number): number => {
  const length = frameEdgeLength(rect, edge);
  if (k <= 0 || k >= segments) {
    return k <= 0 ? 0 : length;
  }
  const start = edge === 0 ? rect.x : edge === 1 ? rect.y : edge === 2 ? rect.x + rect.width : rect.y + rect.height;
  const sign = edge < 2 ? 1 : -1;
  return sign * (Math.round(start + (sign * k * length) / segments) - start);
};

/** `inner` and `outer` are offsets from the core line's centre, positive outward. */
const frameBandRect = (
  ctx: Ctx,
  rect: Rect,
  edge: number,
  from: number,
  to: number,
  inner: number,
  outer: number
): void => {
  if (to <= from) {
    return;
  }
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  if (edge === 0) {
    ctx.rect(rect.x + from, rect.y - outer, to - from, outer - inner);
  } else if (edge === 1) {
    ctx.rect(right + inner, rect.y + from, outer - inner, to - from);
  } else if (edge === 2) {
    ctx.rect(right - to, bottom + inner, to - from, outer - inner);
  } else {
    ctx.rect(rect.x - outer, bottom - to, outer - inner, to - from);
  }
};

/** Corner dashes extend by the band's outer offset so both edges tile one whole L with no seam. */
const traceBboxFrameBand = (ctx: Ctx, frame: BboxFrameGeometry, dash: number, inner: number, outer: number): void => {
  const { rect } = frame;
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  const reach = Math.max(Math.abs(inner), Math.abs(outer));
  ctx.beginPath();
  for (let edge = 0; edge < 4; edge++) {
    const across = edge === 0 ? rect.y : edge === 1 ? right : edge === 2 ? bottom : rect.x;
    if (across + reach < 0 || across - reach > (edge % 2 === 0 ? frame.height : frame.width)) {
      continue;
    }
    const near = edge === 0 ? -rect.x : edge === 1 ? -rect.y : edge === 2 ? right - frame.width : bottom - frame.height;
    const far = edge === 0 ? frame.width - rect.x : edge === 1 ? frame.height - rect.y : edge === 2 ? right : bottom;
    const segments = frameEdgeSegments(rect, edge, dash);
    const step = frameEdgeLength(rect, edge) / segments;
    const first = Math.max(0, Math.floor((near - reach) / step / 2) * 2 - 2);
    const last = Math.min(segments, Math.ceil((far + reach) / step) + 2);
    for (let k = first; k < last; k += 2) {
      const from = frameBoundary(rect, edge, k, segments) - (k === 0 ? outer : 0);
      const to = frameBoundary(rect, edge, k + 1, segments) + (k + 1 === segments ? outer : 0);
      frameBandRect(ctx, rect, edge, from, to, inner, outer);
    }
  }
};

const drawBboxFrame = (ctx: Ctx, frame: BboxFrameGeometry, dpr: number): void => {
  const { hairline } = frame;
  const dash = BBOX_FRAME_DASH_PX * dpr;
  const half = hairline / 2;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = BBOX_CASING_OUTSIDE;
  traceBboxFrameBand(ctx, frame, dash, half, half * 3);
  ctx.fill();
  ctx.fillStyle = BBOX_CASING_INSIDE;
  traceBboxFrameBand(ctx, frame, dash, -half * 3, -half);
  ctx.fill();
  ctx.fillStyle = BBOX_COLOR;
  traceBboxFrameBand(ctx, frame, dash, -half, half);
  ctx.fill();
  ctx.restore();
};

const traceBboxHandles = (ctx: Ctx, rect: Rect, odd: boolean, half: number): void => {
  ctx.beginPath();
  for (const handle of BBOX_HANDLES) {
    const point = bboxHandlePoint(rect, handle);
    const x = snapCentre(point.x, odd, 1);
    const y = snapCentre(point.y, odd, 1);
    ctx.rect(x - half, y - half, half * 2, half * 2);
  }
};

/** Each band is one batched fill, so handles a small frame overlaps merge into one shape. */
const drawBboxHandles = (ctx: Ctx, frame: BboxFrameGeometry, dpr: number): void => {
  const { hairline, rect } = frame;
  const odd = hairline % 2 === 1;
  const parity = odd ? 1 : 0;
  const faceHalf = Math.max(1, Math.round((BBOX_HANDLE_FACE_PX * dpr - parity) / 2) * 2 + parity) / 2;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = BBOX_CASING_OUTSIDE;
  traceBboxHandles(ctx, rect, odd, faceHalf + hairline * 2);
  ctx.fill();
  ctx.fillStyle = BBOX_COLOR;
  traceBboxHandles(ctx, rect, odd, faceHalf + hairline);
  ctx.fill();
  ctx.fillStyle = BBOX_HANDLE_FILL;
  traceBboxHandles(ctx, rect, odd, faceHalf);
  ctx.fill();
  ctx.restore();
};

/** Clears the backing store, then redraws the overlay in CSS pixels under the state's device-pixel ratio. */
export const renderOverlay = (
  target: RasterSurface,
  state: OverlayState,
  renderVectors?: (ctx: RasterSurface['ctx'], state: OverlayState) => void
): void => {
  const ctx = target.ctx;
  const base: Mat2d = { a: state.dpr, b: 0, c: 0, d: state.dpr, e: 0, f: 0 };

  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, target.width, target.height);
  ctx.setTransform(base.a, base.b, base.c, base.d, base.e, base.f);

  drawBboxOverlayShade(ctx, target, state);

  if (state.showGrid) {
    drawGrid(ctx, state);
  }

  drawRuleOfThirds(ctx, state);

  drawStagedRegions(ctx, state);

  // Active-tool handles stay editable even when the passive frame is hidden.
  const bboxFrame = bboxFrameGeometry(target, state);
  if (state.showBbox ?? true) {
    drawBboxFrame(ctx, bboxFrame, state.dpr);
  }

  drawSamPreview(ctx, state, base);
  drawSamGeometry(ctx, state);
  renderVectors?.(ctx, state);
  drawLayerOutline(ctx, state);
  if (state.bboxHandles) {
    drawBboxHandles(ctx, bboxFrame, state.dpr);
  }
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
