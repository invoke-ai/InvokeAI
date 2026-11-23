/** Legacy opacity strokes use max coverage per stamp; source-over would darken overlapping stamps. */
interface Coordinate {
  x: number;
  y: number;
}
interface CoordinateWithPressure extends Coordinate {
  pressure: number;
}
interface RgbaColor {
  r: number;
  g: number;
  b: number;
  a: number;
}
interface PressureStrokeRenderOp {
  type: 'dot';
  x: number;
  y: number;
  radius: number;
  color: RgbaColor;
}
const MIN_PRESSURE_FACTOR = 0.05;
const PRESSURE_SMOOTHING_CENTER_WEIGHT = 0.6;
const PRESSURE_SMOOTHING_NEIGHBOR_WEIGHT = 0.2;
const OPACITY_SMOOTHING_CENTER_WEIGHT = 0.5;
const OPACITY_SMOOTHING_NEIGHBOR_WEIGHT = 0.25;
const OPACITY_SMOOTHING_PASSES = 2;
const OPACITY_ENDPOINT_CENTER_WEIGHT = 0.25;
const OPACITY_ENDPOINT_NEIGHBOR_WEIGHT = 0.75;
const OPACITY_DELTA_STEP_SCALE = 12;
const MIN_OPACITY_STAMP_SPACING_PX = 1;
const OPACITY_STAMP_SPACING_STROKE_WIDTH_SCALE = 0.03;
const MAX_OPACITY_STAMPS_PER_SEGMENT = 128;

const reserveWork = (budget: { workUnits: number }, units: number): void => {
  if (units > budget.workUnits) {
    throw new Error('legacy-conversion-work-limit');
  }
  budget.workUnits -= units;
};

const clampPressure = (pressure: number): number => Math.min(Math.max(pressure, 0), 1);

const getPressureWidthFactor = (pressure: number, pressureAffectsWidth: boolean): number => {
  if (!pressureAffectsWidth) {
    return 1;
  }

  return Math.max(clampPressure(pressure), MIN_PRESSURE_FACTOR);
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
  reserveStamp: (radius: number) => void;
}): PressureStrokeRenderOp[] => {
  const { pressurePoints, strokeWidth, color, pressureAffectsWidth, includeLeadingDot, reserveStamp } = arg;

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
    reserveStamp((strokeWidth * widthFactor) / 2);

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

    for (let sampleIndex = sampleStart; sampleIndex <= stampCount; sampleIndex++) {
      const t = sampleIndex / stampCount;
      const samplePoint = lerpPointWithPressure(prevPoint, nextPoint, t);
      const widthFactor = getPressureWidthFactor(samplePoint.pressure, pressureAffectsWidth);
      const opacityFactor = clampPressure(samplePoint.pressure);

      if (ops.length >= 100000) {
        throw new Error('legacy-stroke-too-large');
      }
      reserveStamp((strokeWidth * widthFactor) / 2);
      ops.push({
        type: 'dot',
        x: samplePoint.x,
        y: samplePoint.y,
        radius: (strokeWidth * widthFactor) / 2,
        color: scaleColorOpacity(color, opacityFactor),
      });
    }
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

export const prepareLegacyOpacityStroke = (
  options: {
    points: number[];
    strokeWidth: number;
    color: RgbaColor;
    pressureAffectsWidth: boolean;
  },
  bounds: { width: number; height: number },
  budget: { workUnits: number }
) => {
  // Reserve output surface, ImageData, put/draw and the initial stamp surface before inspecting points.
  reserveWork(budget, 4 * bounds.width * bounds.height + 1);
  // Twenty units per raw point cover chunking, expanded geometry, smoothing passes and segment planning.
  reserveWork(budget, Math.ceil(options.points.length / 3) * 20);
  const pressurePoints = smoothOpacityEndpointPressurePoints(
    smoothOpacityPressurePoints(smoothStrokeGeometryPoints(smoothPressurePoints(chunkPressurePoints(options.points))))
  );
  let sampledPixels = 0;
  let stampSize = 1;
  const ops = buildOpacityStampRenderOps({
    ...options,
    pressurePoints,
    includeLeadingDot: true,
    reserveStamp: (radius) => {
      const size = Math.ceil(radius * 2 + 4);
      const pixels = size * size;
      if (sampledPixels + pixels > 16 * 1024 * 1024) {
        throw new Error('legacy-stroke-too-large');
      }
      // One planning unit, each clear/read/coverage merge, and any growth of the stamp surface.
      reserveWork(budget, 1 + 3 * pixels + Math.max(0, pixels - stampSize * stampSize));
      sampledPixels += pixels;
      stampSize = Math.max(stampSize, size);
    },
  });
  return { ops, stampSize };
};

export const drawLegacyOpacityStroke = (
  ctx: OffscreenCanvasRenderingContext2D,
  { ops, stampSize }: ReturnType<typeof prepareLegacyOpacityStroke>,
  bounds: { x: number; y: number; width: number; height: number }
): void => {
  const patch = new OffscreenCanvas(bounds.width, bounds.height);
  // Sampling must fit a whole stamp even when the output is only a thin row or column.
  let stamp: OffscreenCanvas | null = null;
  try {
    stamp = new OffscreenCanvas(stampSize, stampSize);
    const patchCtx = patch.getContext('2d');
    const stampCtx = stamp.getContext('2d', { willReadFrequently: true });
    if (!patchCtx || !stampCtx) {
      throw new Error('canvas-unavailable');
    }
    const output = patchCtx.createImageData(bounds.width, bounds.height);
    for (const op of ops) {
      const x = Math.floor(op.x - op.radius - 2),
        y = Math.floor(op.y - op.radius - 2);
      const width = Math.ceil(op.radius * 2 + 4),
        height = width;
      stampCtx.clearRect(0, 0, width, height);
      stampCtx.beginPath();
      stampCtx.arc(op.x - x, op.y - y, op.radius, 0, Math.PI * 2);
      stampCtx.fill();
      const pixels = stampCtx.getImageData(0, 0, width, height).data;
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          const targetX = x + col - bounds.x,
            targetY = y + row - bounds.y;
          if (targetX < 0 || targetY < 0 || targetX >= bounds.width || targetY >= bounds.height) {
            continue;
          }
          const index = (targetY * bounds.width + targetX) * 4;
          const alpha = Math.round(
            (pixels[(row * width + col) * 4 + 3]! * Math.round(clampPressure(op.color.a) * 255)) / 255
          );
          if (alpha > output.data[index + 3]!) {
            output.data[index] = op.color.r;
            output.data[index + 1] = op.color.g;
            output.data[index + 2] = op.color.b;
            output.data[index + 3] = alpha;
          }
        }
      }
    }
    patchCtx.putImageData(output, 0, 0);
    ctx.drawImage(patch, bounds.x, bounds.y);
  } finally {
    patch.width = 1;
    patch.height = 1;
    if (stamp) {
      stamp.width = 1;
      stamp.height = 1;
    }
  }
};
