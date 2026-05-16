import type { Coordinate, CoordinateWithPressure, Rect, RgbaColor } from 'features/controlLayers/store/types';

const MIN_PRESSURE_FACTOR = 0.05;
const PRESSURE_STROKE_RENDER_PADDING_PX = 2;

export type PressureStrokeRenderOp =
  | {
      type: 'dot';
      x: number;
      y: number;
      radius: number;
      color: RgbaColor;
    }
  | {
      type: 'segment';
      from: Coordinate;
      to: Coordinate;
      width: number;
      color: RgbaColor;
    };

const clampPressure = (pressure: number): number => Math.min(Math.max(pressure, 0), 1);

const getPressureWidthFactor = (pressure: number, pressureAffectsWidth: boolean): number => {
  if (!pressureAffectsWidth) {
    return 1;
  }

  return Math.max(clampPressure(pressure), MIN_PRESSURE_FACTOR);
};

const getPressureOpacityFactor = (pressure: number, pressureAffectsOpacity: boolean): number => {
  if (!pressureAffectsOpacity) {
    return 1;
  }

  return clampPressure(pressure);
};

const scaleColorOpacity = (color: RgbaColor, opacityFactor: number): RgbaColor => ({
  ...color,
  a: Math.min(Math.max(color.a * opacityFactor, 0), 1),
});

const chunkPressurePoints = (points: number[]): CoordinateWithPressure[] => {
  const chunked: CoordinateWithPressure[] = [];

  for (let i = 0; i < points.length; i += 3) {
    const x = points[i];
    const y = points[i + 1];
    const pressure = points[i + 2];

    if (x === undefined || y === undefined || pressure === undefined) {
      continue;
    }

    chunked.push({ x, y, pressure });
  }

  return chunked;
};

export const getShouldUsePressureForBrush = (
  pressureAffectsWidth: boolean,
  pressureAffectsOpacity: boolean
): boolean => pressureAffectsWidth || pressureAffectsOpacity;

export const getShouldUsePressureForEraser = (pressureAffectsWidth: boolean): boolean => pressureAffectsWidth;

const getPressureStrokeRenderOpBounds = (renderOp: PressureStrokeRenderOp): Rect | null => {
  if (renderOp.color.a <= 0) {
    return null;
  }

  if (renderOp.type === 'dot') {
    if (renderOp.radius <= 0) {
      return null;
    }

    const x = Math.floor(renderOp.x - renderOp.radius - PRESSURE_STROKE_RENDER_PADDING_PX);
    const y = Math.floor(renderOp.y - renderOp.radius - PRESSURE_STROKE_RENDER_PADDING_PX);
    const maxX = Math.ceil(renderOp.x + renderOp.radius + PRESSURE_STROKE_RENDER_PADDING_PX);
    const maxY = Math.ceil(renderOp.y + renderOp.radius + PRESSURE_STROKE_RENDER_PADDING_PX);

    return {
      x,
      y,
      width: maxX - x,
      height: maxY - y,
    };
  }

  if (renderOp.width <= 0) {
    return null;
  }

  const radius = renderOp.width / 2;
  const x = Math.floor(Math.min(renderOp.from.x, renderOp.to.x) - radius - PRESSURE_STROKE_RENDER_PADDING_PX);
  const y = Math.floor(Math.min(renderOp.from.y, renderOp.to.y) - radius - PRESSURE_STROKE_RENDER_PADDING_PX);
  const maxX = Math.ceil(Math.max(renderOp.from.x, renderOp.to.x) + radius + PRESSURE_STROKE_RENDER_PADDING_PX);
  const maxY = Math.ceil(Math.max(renderOp.from.y, renderOp.to.y) + radius + PRESSURE_STROKE_RENDER_PADDING_PX);

  return {
    x,
    y,
    width: maxX - x,
    height: maxY - y,
  };
};

export const getPressureStrokeRenderBounds = (renderOps: PressureStrokeRenderOp[]): Rect | null => {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const renderOp of renderOps) {
    const bounds = getPressureStrokeRenderOpBounds(renderOp);

    if (!bounds) {
      continue;
    }

    minX = Math.min(minX, bounds.x);
    minY = Math.min(minY, bounds.y);
    maxX = Math.max(maxX, bounds.x + bounds.width);
    maxY = Math.max(maxY, bounds.y + bounds.height);
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null;
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY,
  };
};

const getCanvasContext = (
  canvas: HTMLCanvasElement,
  width: number,
  height: number
): CanvasRenderingContext2D | null => {
  if (canvas.width !== width) {
    canvas.width = width;
  }

  if (canvas.height !== height) {
    canvas.height = height;
  }

  const ctx = canvas.getContext('2d');

  if (!ctx) {
    return null;
  }

  ctx.clearRect(0, 0, width, height);
  ctx.imageSmoothingEnabled = true;
  ctx.fillStyle = 'rgba(0, 0, 0, 1)';
  ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  return ctx;
};

export const renderPressureStrokeToCanvas = (
  renderOps: PressureStrokeRenderOp[]
): { canvas: HTMLCanvasElement; x: number; y: number } | null => {
  const bounds = getPressureStrokeRenderBounds(renderOps);

  if (!bounds || bounds.width <= 0 || bounds.height <= 0) {
    return null;
  }

  const canvas = document.createElement('canvas');
  const ctx = getCanvasContext(canvas, bounds.width, bounds.height);

  if (!ctx) {
    return null;
  }

  const imageData = ctx.createImageData(bounds.width, bounds.height);
  const output = imageData.data;
  const patchCanvas = document.createElement('canvas');

  for (const renderOp of renderOps) {
    const patchBounds = getPressureStrokeRenderOpBounds(renderOp);
    const opacityByte = Math.round(clampPressure(renderOp.color.a) * 255);

    if (!patchBounds || opacityByte === 0) {
      continue;
    }

    const patchCtx = getCanvasContext(patchCanvas, patchBounds.width, patchBounds.height);

    if (!patchCtx) {
      return null;
    }

    if (renderOp.type === 'dot') {
      patchCtx.beginPath();
      patchCtx.arc(renderOp.x - patchBounds.x, renderOp.y - patchBounds.y, renderOp.radius, 0, Math.PI * 2);
      patchCtx.fill();
    } else {
      patchCtx.beginPath();
      patchCtx.lineWidth = renderOp.width;
      patchCtx.moveTo(renderOp.from.x - patchBounds.x, renderOp.from.y - patchBounds.y);
      patchCtx.lineTo(renderOp.to.x - patchBounds.x, renderOp.to.y - patchBounds.y);
      patchCtx.stroke();
    }

    const patchData = patchCtx.getImageData(0, 0, patchBounds.width, patchBounds.height).data;
    const targetOffsetX = patchBounds.x - bounds.x;
    const targetOffsetY = patchBounds.y - bounds.y;

    for (let patchY = 0; patchY < patchBounds.height; patchY++) {
      for (let patchX = 0; patchX < patchBounds.width; patchX++) {
        const patchIndex = (patchY * patchBounds.width + patchX) * 4;
        const coverageAlpha = patchData[patchIndex + 3];

        if (!coverageAlpha) {
          continue;
        }

        const targetX = targetOffsetX + patchX;
        const targetY = targetOffsetY + patchY;
        const targetIndex = (targetY * bounds.width + targetX) * 4;
        const candidateAlpha = Math.round((coverageAlpha * opacityByte) / 255);
        const currentAlpha = output[targetIndex + 3] ?? 0;

        if (candidateAlpha <= currentAlpha) {
          continue;
        }

        output[targetIndex] = renderOp.color.r;
        output[targetIndex + 1] = renderOp.color.g;
        output[targetIndex + 2] = renderOp.color.b;
        output[targetIndex + 3] = candidateAlpha;
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);

  return {
    canvas,
    x: bounds.x,
    y: bounds.y,
  };
};

export const getPressureStrokeRenderOps = (arg: {
  points: number[];
  strokeWidth: number;
  color: RgbaColor;
  pressureAffectsWidth: boolean;
  pressureAffectsOpacity: boolean;
}): PressureStrokeRenderOp[] => {
  const { points, strokeWidth, color, pressureAffectsWidth, pressureAffectsOpacity } = arg;
  const pressurePoints = chunkPressurePoints(points);

  if (pressurePoints.length === 0) {
    return [];
  }

  if (pressurePoints.length === 1) {
    const point = pressurePoints[0];

    if (!point) {
      return [];
    }

    const widthFactor = getPressureWidthFactor(point.pressure, pressureAffectsWidth);
    const opacityFactor = getPressureOpacityFactor(point.pressure, pressureAffectsOpacity);

    return [
      {
        type: 'dot',
        x: point.x,
        y: point.y,
        radius: (strokeWidth * widthFactor) / 2,
        color: scaleColorOpacity(color, opacityFactor),
      },
    ];
  }

  const ops: PressureStrokeRenderOp[] = [];

  for (let i = 1; i < pressurePoints.length; i++) {
    const prevPoint = pressurePoints[i - 1];
    const nextPoint = pressurePoints[i];

    if (!prevPoint || !nextPoint) {
      continue;
    }

    const segmentPressure = (prevPoint.pressure + nextPoint.pressure) / 2;
    const widthFactor = getPressureWidthFactor(segmentPressure, pressureAffectsWidth);
    const opacityFactor = getPressureOpacityFactor(segmentPressure, pressureAffectsOpacity);

    ops.push({
      type: 'segment',
      from: { x: prevPoint.x, y: prevPoint.y },
      to: { x: nextPoint.x, y: nextPoint.y },
      width: strokeWidth * widthFactor,
      color: scaleColorOpacity(color, opacityFactor),
    });
  }

  return ops;
};
