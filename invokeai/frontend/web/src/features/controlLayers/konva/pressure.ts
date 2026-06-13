import type { Coordinate, CoordinateWithPressure, Rect, RgbaColor } from 'features/controlLayers/store/types';

const MIN_PRESSURE_FACTOR = 0.05;
const PRESSURE_STROKE_RENDER_PADDING_PX = 2;
const PRESSURE_SMOOTHING_CENTER_WEIGHT = 0.6;
const PRESSURE_SMOOTHING_NEIGHBOR_WEIGHT = 0.2;
const OPACITY_SMOOTHING_CENTER_WEIGHT = 0.5;
const OPACITY_SMOOTHING_NEIGHBOR_WEIGHT = 0.25;
const OPACITY_SMOOTHING_PASSES = 2;
const OPACITY_ENDPOINT_CENTER_WEIGHT = 0.25;
const OPACITY_ENDPOINT_NEIGHBOR_WEIGHT = 0.75;
const MIN_RENDER_SEGMENT_LENGTH_PX = 4;
const RENDER_SEGMENT_LENGTH_STROKE_WIDTH_SCALE = 0.25;
const PRESSURE_DELTA_STEP_SCALE = 8;
const OPACITY_DELTA_STEP_SCALE = 12;
const MAX_RENDER_SUBSEGMENTS_PER_SEGMENT = 24;
const MIN_OPACITY_STAMP_SPACING_PX = 1;
const OPACITY_STAMP_SPACING_STROKE_WIDTH_SCALE = 0.03;
const MAX_OPACITY_STAMPS_PER_SEGMENT = 128;
const INCREMENTAL_PREVIEW_POINT_OVERLAP = 2;

type PressureStrokeRenderOp =
  | {
      type: 'dot';
      x: number;
      y: number;
      radius: number;
      color: RgbaColor;
      strokeDistance?: number;
    }
  | {
      type: 'segment';
      from: Coordinate;
      to: Coordinate;
      width: number;
      color: RgbaColor;
    };

export type PressureStrokeCanvasTarget = {
  canvas: HTMLCanvasElement;
  x: number;
  y: number;
  imageData: ImageData;
};

type PressureStrokeOpacityDotRenderOp = Extract<PressureStrokeRenderOp, { type: 'dot' }> & {
  strokeDistance: number;
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

const smoothPressurePointsWithWeights = (
  points: CoordinateWithPressure[],
  centerWeight: number,
  neighborWeight: number,
  passes: number = 1
): CoordinateWithPressure[] => {
  if (points.length <= 2) {
    return points;
  }

  let smoothedPoints = points;

  for (let pass = 0; pass < passes; pass++) {
    smoothedPoints = smoothedPoints.map((point, index) => {
      if (index === 0 || index === smoothedPoints.length - 1) {
        return point;
      }

      const prevPoint = smoothedPoints[index - 1];
      const nextPoint = smoothedPoints[index + 1];

      if (!prevPoint || !nextPoint) {
        return point;
      }

      return {
        ...point,
        pressure: clampPressure(
          prevPoint.pressure * neighborWeight + point.pressure * centerWeight + nextPoint.pressure * neighborWeight
        ),
      };
    });
  }

  return smoothedPoints;
};

const smoothPressurePoints = (points: CoordinateWithPressure[]): CoordinateWithPressure[] =>
  smoothPressurePointsWithWeights(points, PRESSURE_SMOOTHING_CENTER_WEIGHT, PRESSURE_SMOOTHING_NEIGHBOR_WEIGHT);

const smoothStrokeGeometryPoints = (points: CoordinateWithPressure[]): CoordinateWithPressure[] => {
  if (points.length <= 2) {
    return points;
  }

  const smoothed: CoordinateWithPressure[] = [];
  const firstPoint = points[0];
  const lastPoint = points.at(-1);

  if (!firstPoint || !lastPoint) {
    return points;
  }

  smoothed.push(firstPoint);

  for (let i = 0; i < points.length - 1; i++) {
    const point = points[i];
    const nextPoint = points[i + 1];

    if (!point || !nextPoint) {
      continue;
    }

    smoothed.push({
      x: point.x * 0.75 + nextPoint.x * 0.25,
      y: point.y * 0.75 + nextPoint.y * 0.25,
      pressure: clampPressure(point.pressure * 0.75 + nextPoint.pressure * 0.25),
    });
    smoothed.push({
      x: point.x * 0.25 + nextPoint.x * 0.75,
      y: point.y * 0.25 + nextPoint.y * 0.75,
      pressure: clampPressure(point.pressure * 0.25 + nextPoint.pressure * 0.75),
    });
  }

  smoothed.push(lastPoint);

  return smoothed;
};

const smoothOpacityPressurePoints = (points: CoordinateWithPressure[]): CoordinateWithPressure[] =>
  smoothPressurePointsWithWeights(
    points,
    OPACITY_SMOOTHING_CENTER_WEIGHT,
    OPACITY_SMOOTHING_NEIGHBOR_WEIGHT,
    OPACITY_SMOOTHING_PASSES
  );

const smoothOpacityEndpointPressurePoints = (points: CoordinateWithPressure[]): CoordinateWithPressure[] => {
  if (points.length <= 1) {
    return points;
  }

  const smoothedPoints = [...points];
  const firstPoint = smoothedPoints[0];
  const secondPoint = smoothedPoints[1];
  const lastPoint = smoothedPoints.at(-1);
  const penultimatePoint = smoothedPoints.at(-2);

  if (firstPoint && secondPoint) {
    smoothedPoints[0] = {
      ...firstPoint,
      pressure: clampPressure(
        firstPoint.pressure * OPACITY_ENDPOINT_CENTER_WEIGHT + secondPoint.pressure * OPACITY_ENDPOINT_NEIGHBOR_WEIGHT
      ),
    };
  }

  if (lastPoint && penultimatePoint) {
    smoothedPoints[smoothedPoints.length - 1] = {
      ...lastPoint,
      pressure: clampPressure(
        lastPoint.pressure * OPACITY_ENDPOINT_CENTER_WEIGHT +
          penultimatePoint.pressure * OPACITY_ENDPOINT_NEIGHBOR_WEIGHT
      ),
    };
  }

  return smoothedPoints;
};

const buildOpacityStampRenderOps = (arg: {
  pressurePoints: CoordinateWithPressure[];
  strokeWidth: number;
  color: RgbaColor;
  pressureAffectsWidth: boolean;
  includeLeadingDot: boolean;
}): PressureStrokeRenderOp[] => {
  const { pressurePoints, strokeWidth, color, pressureAffectsWidth, includeLeadingDot } = arg;

  if (pressurePoints.length === 0) {
    return [];
  }

  if (pressurePoints.length === 1) {
    if (!includeLeadingDot) {
      return [];
    }

    const point = pressurePoints[0];

    if (!point) {
      return [];
    }

    const widthFactor = getPressureWidthFactor(point.pressure, pressureAffectsWidth);
    const opacityFactor = clampPressure(point.pressure);

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
  let strokeDistance = 0;
  for (let i = 1; i < pressurePoints.length; i++) {
    const prevPoint = pressurePoints[i - 1];
    const nextPoint = pressurePoints[i];

    if (!prevPoint || !nextPoint) {
      continue;
    }

    const distance = Math.hypot(nextPoint.x - prevPoint.x, nextPoint.y - prevPoint.y);
    const averageWidthFactor =
      (getPressureWidthFactor(prevPoint.pressure, pressureAffectsWidth) +
        getPressureWidthFactor(nextPoint.pressure, pressureAffectsWidth)) /
      2;
    const stampSpacing = Math.max(
      MIN_OPACITY_STAMP_SPACING_PX,
      strokeWidth * averageWidthFactor * OPACITY_STAMP_SPACING_STROKE_WIDTH_SCALE
    );
    const stampCount = Math.min(
      MAX_OPACITY_STAMPS_PER_SEGMENT,
      Math.max(
        1,
        Math.ceil(distance / stampSpacing),
        Math.ceil(Math.abs(nextPoint.pressure - prevPoint.pressure) * OPACITY_DELTA_STEP_SCALE)
      )
    );
    const sampleStart = includeLeadingDot && i === 1 ? 0 : 1;
    const segmentStartDistance = strokeDistance;

    for (let sampleIndex = sampleStart; sampleIndex <= stampCount; sampleIndex++) {
      const t = sampleIndex / stampCount;
      const samplePoint = lerpPointWithPressure(prevPoint, nextPoint, t);
      const widthFactor = getPressureWidthFactor(samplePoint.pressure, pressureAffectsWidth);
      const opacityFactor = clampPressure(samplePoint.pressure);

      ops.push({
        type: 'dot',
        x: samplePoint.x,
        y: samplePoint.y,
        radius: (strokeWidth * widthFactor) / 2,
        color: scaleColorOpacity(color, opacityFactor),
        strokeDistance: segmentStartDistance + distance * t,
      });
    }

    strokeDistance += distance;
  }

  return ops;
};

const lerp = (from: number, to: number, t: number): number => from + (to - from) * t;

const lerpPointWithPressure = (
  from: CoordinateWithPressure,
  to: CoordinateWithPressure,
  t: number
): CoordinateWithPressure => ({
  x: lerp(from.x, to.x, t),
  y: lerp(from.y, to.y, t),
  pressure: lerp(from.pressure, to.pressure, t),
});

export const getShouldUsePressureForBrush = (pressureAffectsWidth: boolean, pressureAffectsOpacity: boolean): boolean =>
  pressureAffectsWidth || pressureAffectsOpacity;

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

const mergeRects = (a: Rect, b: Rect): Rect => {
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const maxX = Math.max(a.x + a.width, b.x + b.width);
  const maxY = Math.max(a.y + a.height, b.y + b.height);

  return {
    x,
    y,
    width: maxX - x,
    height: maxY - y,
  };
};

const isOpacityDotRenderOp = (renderOp: PressureStrokeRenderOp): renderOp is PressureStrokeOpacityDotRenderOp =>
  renderOp.type === 'dot' && typeof renderOp.strokeDistance === 'number';

const compositeSourceOverAlphaByte = (currentAlpha: number, candidateAlpha: number): number =>
  Math.round(currentAlpha + (candidateAlpha * (255 - currentAlpha)) / 255);

export const mergeOpacityDotAlphaAtPixel = (arg: {
  currentAlpha: number;
  candidateAlpha: number;
  lastStrokeDistance: number;
  strokeDistance: number;
  lastRadius: number;
  radius: number;
}): { alpha: number; lastStrokeDistance: number; lastRadius: number } => {
  const { currentAlpha, candidateAlpha, lastStrokeDistance, strokeDistance, lastRadius, radius } = arg;

  if (currentAlpha <= 0 || !Number.isFinite(lastStrokeDistance)) {
    return {
      alpha: candidateAlpha,
      lastStrokeDistance: strokeDistance,
      lastRadius: radius,
    };
  }

  const revisitDistanceThreshold = Math.max(lastRadius, radius) * 2;
  const isSameLocalPass = strokeDistance - lastStrokeDistance <= revisitDistanceThreshold;

  if (isSameLocalPass) {
    return {
      alpha: Math.max(currentAlpha, candidateAlpha),
      lastStrokeDistance: strokeDistance,
      lastRadius: Math.max(lastRadius, radius),
    };
  }

  return {
    alpha: compositeSourceOverAlphaByte(currentAlpha, candidateAlpha),
    lastStrokeDistance: strokeDistance,
    lastRadius: radius,
  };
};

const getCanvasContext = (
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  clear: boolean = false
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

  if (clear) {
    ctx.clearRect(0, 0, width, height);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.fillStyle = 'rgba(0, 0, 0, 1)';
  ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  return ctx;
};

const copyImageData = (source: ImageData, destination: ImageData, offsetX: number, offsetY: number) => {
  const sourceData = source.data;
  const destinationData = destination.data;

  for (let y = 0; y < source.height; y++) {
    const sourceRowStart = y * source.width * 4;
    const destinationRowStart = ((y + offsetY) * destination.width + offsetX) * 4;
    destinationData.set(sourceData.subarray(sourceRowStart, sourceRowStart + source.width * 4), destinationRowStart);
  }
};

const ensurePressureStrokeCanvasTarget = (
  target: PressureStrokeCanvasTarget | null,
  requiredBounds: Rect
): PressureStrokeCanvasTarget | null => {
  if (!target) {
    const canvas = document.createElement('canvas');
    const ctx = getCanvasContext(canvas, requiredBounds.width, requiredBounds.height);

    if (!ctx) {
      return null;
    }

    return {
      canvas,
      x: requiredBounds.x,
      y: requiredBounds.y,
      imageData: ctx.createImageData(requiredBounds.width, requiredBounds.height),
    };
  }

  const currentBounds = {
    x: target.x,
    y: target.y,
    width: target.imageData.width,
    height: target.imageData.height,
  };
  const nextBounds = mergeRects(currentBounds, requiredBounds);

  if (
    nextBounds.x === currentBounds.x &&
    nextBounds.y === currentBounds.y &&
    nextBounds.width === currentBounds.width &&
    nextBounds.height === currentBounds.height
  ) {
    return target;
  }

  const ctx = getCanvasContext(target.canvas, nextBounds.width, nextBounds.height);

  if (!ctx) {
    return null;
  }

  const nextImageData = ctx.createImageData(nextBounds.width, nextBounds.height);
  copyImageData(target.imageData, nextImageData, currentBounds.x - nextBounds.x, currentBounds.y - nextBounds.y);

  target.x = nextBounds.x;
  target.y = nextBounds.y;
  target.imageData = nextImageData;

  return target;
};

export const appendPressureStrokeRenderOpsToCanvas = (
  target: PressureStrokeCanvasTarget | null,
  renderOps: PressureStrokeRenderOp[]
): PressureStrokeCanvasTarget | null => {
  const renderBounds = getPressureStrokeRenderBounds(renderOps);

  if (!renderBounds || renderBounds.width <= 0 || renderBounds.height <= 0) {
    return target;
  }

  const nextTarget = ensurePressureStrokeCanvasTarget(target, renderBounds);

  if (!nextTarget) {
    return null;
  }

  const ctx = getCanvasContext(nextTarget.canvas, nextTarget.imageData.width, nextTarget.imageData.height);

  if (!ctx) {
    return null;
  }

  const output = nextTarget.imageData.data;
  const patchCanvas = document.createElement('canvas');
  for (const renderOp of renderOps) {
    const patchBounds = getPressureStrokeRenderOpBounds(renderOp);
    const opacityByte = Math.round(clampPressure(renderOp.color.a) * 255);

    if (!patchBounds || opacityByte === 0) {
      continue;
    }

    const patchCtx = getCanvasContext(patchCanvas, patchBounds.width, patchBounds.height, true);

    if (!patchCtx) {
      return null;
    }

    if (renderOp.type === 'dot') {
      patchCtx.beginPath();
      patchCtx.arc(renderOp.x - patchBounds.x, renderOp.y - patchBounds.y, renderOp.radius, 0, Math.PI * 2);
      patchCtx.fill();
    } else {
      patchCtx.lineCap = 'round';
      patchCtx.beginPath();
      patchCtx.lineWidth = renderOp.width;
      patchCtx.moveTo(renderOp.from.x - patchBounds.x, renderOp.from.y - patchBounds.y);
      patchCtx.lineTo(renderOp.to.x - patchBounds.x, renderOp.to.y - patchBounds.y);
      patchCtx.stroke();
    }

    const patchData = patchCtx.getImageData(0, 0, patchBounds.width, patchBounds.height).data;
    const targetOffsetX = patchBounds.x - nextTarget.x;
    const targetOffsetY = patchBounds.y - nextTarget.y;

    for (let patchY = 0; patchY < patchBounds.height; patchY++) {
      for (let patchX = 0; patchX < patchBounds.width; patchX++) {
        const patchIndex = (patchY * patchBounds.width + patchX) * 4;
        const coverageAlpha = patchData[patchIndex + 3];

        if (!coverageAlpha) {
          continue;
        }

        const targetX = targetOffsetX + patchX;
        const targetY = targetOffsetY + patchY;
        const targetIndex = (targetY * nextTarget.imageData.width + targetX) * 4;
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

  ctx.putImageData(nextTarget.imageData, 0, 0);

  return nextTarget;
};

const renderOpacityStrokeDotsToCanvas = (
  renderOps: PressureStrokeOpacityDotRenderOp[]
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
  const lastStrokeDistances = new Float32Array(bounds.width * bounds.height);
  lastStrokeDistances.fill(Number.NEGATIVE_INFINITY);
  const lastRadii = new Float32Array(bounds.width * bounds.height);
  const patchCanvas = document.createElement('canvas');

  for (const renderOp of renderOps) {
    const patchBounds = getPressureStrokeRenderOpBounds(renderOp);
    const opacityByte = Math.round(clampPressure(renderOp.color.a) * 255);

    if (!patchBounds || opacityByte === 0) {
      continue;
    }

    const patchCtx = getCanvasContext(patchCanvas, patchBounds.width, patchBounds.height, true);

    if (!patchCtx) {
      return null;
    }

    patchCtx.beginPath();
    patchCtx.arc(renderOp.x - patchBounds.x, renderOp.y - patchBounds.y, renderOp.radius, 0, Math.PI * 2);
    patchCtx.fill();

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
        const pixelIndex = targetY * bounds.width + targetX;
        const targetIndex = pixelIndex * 4;
        const candidateAlpha = Math.round((coverageAlpha * opacityByte) / 255);

        const merged = mergeOpacityDotAlphaAtPixel({
          currentAlpha: output[targetIndex + 3] ?? 0,
          candidateAlpha,
          lastStrokeDistance: lastStrokeDistances[pixelIndex] ?? Number.NEGATIVE_INFINITY,
          strokeDistance: renderOp.strokeDistance,
          lastRadius: lastRadii[pixelIndex] ?? 0,
          radius: renderOp.radius,
        });

        output[targetIndex] = renderOp.color.r;
        output[targetIndex + 1] = renderOp.color.g;
        output[targetIndex + 2] = renderOp.color.b;
        output[targetIndex + 3] = merged.alpha;
        lastStrokeDistances[pixelIndex] = merged.lastStrokeDistance;
        lastRadii[pixelIndex] = merged.lastRadius;
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

export const renderPressureStrokeToCanvas = (
  renderOps: PressureStrokeRenderOp[]
): { canvas: HTMLCanvasElement; x: number; y: number } | null => {
  if (renderOps.length > 0 && renderOps.every(isOpacityDotRenderOp)) {
    return renderOpacityStrokeDotsToCanvas(renderOps);
  }

  const target = appendPressureStrokeRenderOpsToCanvas(null, renderOps);

  if (!target) {
    return null;
  }

  return {
    canvas: target.canvas,
    x: target.x,
    y: target.y,
  };
};

const buildPressureStrokeRenderOps = (arg: {
  pressurePoints: CoordinateWithPressure[];
  strokeWidth: number;
  color: RgbaColor;
  pressureAffectsWidth: boolean;
  pressureAffectsOpacity: boolean;
  includeLeadingDot: boolean;
  includeTrailingDot: boolean;
}): PressureStrokeRenderOp[] => {
  const {
    pressurePoints,
    strokeWidth,
    color,
    pressureAffectsWidth,
    pressureAffectsOpacity,
    includeLeadingDot,
    includeTrailingDot,
  } = arg;

  if (pressurePoints.length === 0) {
    return [];
  }

  if (pressurePoints.length === 1) {
    if (!includeLeadingDot) {
      return [];
    }

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

  if (pressureAffectsOpacity) {
    return buildOpacityStampRenderOps({
      pressurePoints,
      strokeWidth,
      color,
      pressureAffectsWidth,
      includeLeadingDot,
    });
  }

  const ops: PressureStrokeRenderOp[] = [];
  const targetSegmentLength = Math.max(
    MIN_RENDER_SEGMENT_LENGTH_PX,
    strokeWidth * RENDER_SEGMENT_LENGTH_STROKE_WIDTH_SCALE
  );
  const pressureDeltaStepScale = PRESSURE_DELTA_STEP_SCALE;
  const firstPoint = pressurePoints[0];
  const lastPoint = pressurePoints.at(-1);

  if (includeLeadingDot && firstPoint) {
    const widthFactor = getPressureWidthFactor(firstPoint.pressure, pressureAffectsWidth);
    const opacityFactor = getPressureOpacityFactor(firstPoint.pressure, pressureAffectsOpacity);

    ops.push({
      type: 'dot',
      x: firstPoint.x,
      y: firstPoint.y,
      radius: (strokeWidth * widthFactor) / 2,
      color: scaleColorOpacity(color, opacityFactor),
    });
  }

  for (let i = 1; i < pressurePoints.length; i++) {
    const prevPoint = pressurePoints[i - 1];
    const nextPoint = pressurePoints[i];

    if (!prevPoint || !nextPoint) {
      continue;
    }

    const distance = Math.hypot(nextPoint.x - prevPoint.x, nextPoint.y - prevPoint.y);
    const subsegmentCount = Math.min(
      MAX_RENDER_SUBSEGMENTS_PER_SEGMENT,
      Math.max(
        1,
        Math.ceil(distance / targetSegmentLength),
        Math.ceil(Math.abs(nextPoint.pressure - prevPoint.pressure) * pressureDeltaStepScale)
      )
    );

    for (let subsegmentIndex = 0; subsegmentIndex < subsegmentCount; subsegmentIndex++) {
      const fromT = subsegmentIndex / subsegmentCount;
      const toT = (subsegmentIndex + 1) / subsegmentCount;
      const subsegmentFrom = lerpPointWithPressure(prevPoint, nextPoint, fromT);
      const subsegmentTo = lerpPointWithPressure(prevPoint, nextPoint, toT);
      const segmentPressure = (subsegmentFrom.pressure + subsegmentTo.pressure) / 2;
      const widthFactor = getPressureWidthFactor(segmentPressure, pressureAffectsWidth);
      const opacityFactor = getPressureOpacityFactor(segmentPressure, pressureAffectsOpacity);

      ops.push({
        type: 'segment',
        from: { x: subsegmentFrom.x, y: subsegmentFrom.y },
        to: { x: subsegmentTo.x, y: subsegmentTo.y },
        width: strokeWidth * widthFactor,
        color: scaleColorOpacity(color, opacityFactor),
      });
    }
  }

  if (includeTrailingDot && lastPoint) {
    const widthFactor = getPressureWidthFactor(lastPoint.pressure, pressureAffectsWidth);
    const opacityFactor = getPressureOpacityFactor(lastPoint.pressure, pressureAffectsOpacity);

    ops.push({
      type: 'dot',
      x: lastPoint.x,
      y: lastPoint.y,
      radius: (strokeWidth * widthFactor) / 2,
      color: scaleColorOpacity(color, opacityFactor),
    });
  }

  return ops;
};

export const getPressureStrokeRenderOps = (arg: {
  points: number[];
  strokeWidth: number;
  color: RgbaColor;
  pressureAffectsWidth: boolean;
  pressureAffectsOpacity: boolean;
}): PressureStrokeRenderOp[] => {
  const { points, strokeWidth, color, pressureAffectsWidth, pressureAffectsOpacity } = arg;
  const geometryPoints = smoothStrokeGeometryPoints(smoothPressurePoints(chunkPressurePoints(points)));
  const pressurePoints = pressureAffectsOpacity
    ? smoothOpacityEndpointPressurePoints(smoothOpacityPressurePoints(geometryPoints))
    : geometryPoints;

  return buildPressureStrokeRenderOps({
    pressurePoints,
    strokeWidth,
    color,
    pressureAffectsWidth,
    pressureAffectsOpacity,
    includeLeadingDot: true,
    includeTrailingDot: true,
  });
};

export const getPressureStrokeRenderOpsFromPointIndex = (arg: {
  points: number[];
  strokeWidth: number;
  color: RgbaColor;
  pressureAffectsWidth: boolean;
  pressureAffectsOpacity: boolean;
  startPointIndex: number;
}): PressureStrokeRenderOp[] => {
  const { points, strokeWidth, color, pressureAffectsWidth, pressureAffectsOpacity, startPointIndex } = arg;
  const geometryPoints = smoothStrokeGeometryPoints(
    smoothPressurePoints(
      chunkPressurePoints(points).slice(Math.max(0, startPointIndex - INCREMENTAL_PREVIEW_POINT_OVERLAP))
    )
  );
  const pressurePoints = pressureAffectsOpacity
    ? smoothOpacityEndpointPressurePoints(smoothOpacityPressurePoints(geometryPoints))
    : geometryPoints;

  return buildPressureStrokeRenderOps({
    pressurePoints,
    strokeWidth,
    color,
    pressureAffectsWidth,
    pressureAffectsOpacity,
    includeLeadingDot: false,
    includeTrailingDot: false,
  });
};
