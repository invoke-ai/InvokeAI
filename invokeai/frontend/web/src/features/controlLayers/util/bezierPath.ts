import type { CanvasBezierPointState, Coordinate, Rect } from 'features/controlLayers/store/types';

type RenderableBezierPoint = Pick<CanvasBezierPointState, 'anchor' | 'inHandle' | 'outHandle'>;
type BezierPointHandleType = 'inHandle' | 'outHandle';
export type BezierPointType = CanvasBezierPointState['type'];
type BezierPathSegmentHit = {
  segmentIndex: number;
  t: number;
  point: Coordinate;
  distance: number;
};

type CubicBezierSegment = [Coordinate, Coordinate, Coordinate, Coordinate];

const DEFAULT_BEZIER_PATH_SAMPLES_PER_SEGMENT = 24;
const ELLIPSE_BEZIER_CONTROL_POINT_RATIO = (4 * (Math.sqrt(2) - 1)) / 3;
const BEZIER_FIT_EPSILON = 1e-12;
const SMOOTH_HANDLE_MAX_SEGMENT_RATIO = 2 / 3;
const CENTRIPETAL_CATMULL_ROM_ALPHA = 0.5;

const formatCoordinate = (coordinate: Coordinate) => `${coordinate.x} ${coordinate.y}`;
const getDistance = (a: Coordinate, b: Coordinate) => Math.hypot(a.x - b.x, a.y - b.y);
const lerpCoordinate = (a: Coordinate, b: Coordinate, t: number): Coordinate => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});
const addCoordinate = (a: Coordinate, b: Coordinate): Coordinate => ({ x: a.x + b.x, y: a.y + b.y });
const subtractCoordinate = (a: Coordinate, b: Coordinate): Coordinate => ({ x: a.x - b.x, y: a.y - b.y });
const scaleCoordinate = (coordinate: Coordinate, scale: number): Coordinate => ({
  x: coordinate.x * scale,
  y: coordinate.y * scale,
});
const dotCoordinates = (a: Coordinate, b: Coordinate): number => a.x * b.x + a.y * b.y;
const normalizeCoordinate = (coordinate: Coordinate): Coordinate => {
  const length = Math.hypot(coordinate.x, coordinate.y);
  return length <= BEZIER_FIT_EPSILON ? { x: 0, y: 0 } : scaleCoordinate(coordinate, 1 / length);
};
const getSquaredDistance = (a: Coordinate, b: Coordinate): number => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
const normalizeHandle = (anchor: Coordinate, handle: Coordinate): Coordinate | null =>
  anchor.x === handle.x && anchor.y === handle.y ? null : handle;
const mirrorHandle = (anchor: Coordinate, handle: Coordinate): Coordinate => ({
  x: anchor.x + (anchor.x - handle.x),
  y: anchor.y + (anchor.y - handle.y),
});
const getOppositeHandleType = (handleType: BezierPointHandleType): BezierPointHandleType =>
  handleType === 'inHandle' ? 'outHandle' : 'inHandle';
const getCollinearOppositeHandle = (anchor: Coordinate, handle: Coordinate, length: number): Coordinate | null => {
  const sourceLength = getDistance(anchor, handle);
  if (sourceLength === 0 || length === 0) {
    return null;
  }

  const scale = length / sourceLength;
  return {
    x: anchor.x + (anchor.x - handle.x) * scale,
    y: anchor.y + (anchor.y - handle.y) * scale,
  };
};

const getPreferredHandleType = (
  point: CanvasBezierPointState,
  preferredHandleType?: BezierPointHandleType | null
): BezierPointHandleType | null => {
  if (preferredHandleType && point[preferredHandleType]) {
    return preferredHandleType;
  }
  if (point.outHandle) {
    return 'outHandle';
  }
  if (point.inHandle) {
    return 'inHandle';
  }
  return null;
};

const syncOppositeHandleForPointType = (point: CanvasBezierPointState, sourceHandleType: BezierPointHandleType) => {
  const sourceHandle = point[sourceHandleType];
  const oppositeHandleType = getOppositeHandleType(sourceHandleType);
  if (point.type === 'corner') {
    return;
  }
  if (!sourceHandle) {
    point[oppositeHandleType] = null;
    return;
  }
  if (point.type === 'symmetric') {
    point[oppositeHandleType] = normalizeHandle(point.anchor, mirrorHandle(point.anchor, sourceHandle));
    return;
  }

  const oppositeHandle = point[oppositeHandleType];
  const oppositeLength = oppositeHandle
    ? getDistance(point.anchor, oppositeHandle)
    : getDistance(point.anchor, sourceHandle);
  point[oppositeHandleType] = normalizeHandle(
    point.anchor,
    getCollinearOppositeHandle(point.anchor, sourceHandle, oppositeLength) ?? point.anchor
  );
};

export const setBezierPointType = (
  point: CanvasBezierPointState,
  type: BezierPointType,
  preferredHandleType?: BezierPointHandleType | null
) => {
  point.type = type;
  const sourceHandleType = getPreferredHandleType(point, preferredHandleType);
  if (!sourceHandleType) {
    return;
  }

  syncOppositeHandleForPointType(point, sourceHandleType);
};

export const setBezierPointHandle = (
  point: CanvasBezierPointState,
  handleType: BezierPointHandleType,
  handle: Coordinate
) => {
  point[handleType] = normalizeHandle(point.anchor, handle);
  syncOppositeHandleForPointType(point, handleType);
};

export const getBezierPointPullHandleType = (
  points: Pick<CanvasBezierPointState, 'anchor'>[],
  isClosed: boolean,
  pointIndex: number,
  pointer: Coordinate
): 'inHandle' | 'outHandle' => {
  const point = points[pointIndex];
  if (!point) {
    return 'outHandle';
  }

  if (!isClosed) {
    if (pointIndex === 0) {
      return 'outHandle';
    }
    if (pointIndex === points.length - 1) {
      return 'inHandle';
    }
  }

  const previousPoint = isClosed ? points.at(pointIndex - 1) : points[pointIndex - 1];
  const nextPoint = isClosed ? points[(pointIndex + 1) % points.length] : points[pointIndex + 1];
  if (!previousPoint) {
    return 'outHandle';
  }
  if (!nextPoint) {
    return 'inHandle';
  }

  const dragVector = {
    x: pointer.x - point.anchor.x,
    y: pointer.y - point.anchor.y,
  };
  const dragLength = Math.hypot(dragVector.x, dragVector.y);
  if (dragLength === 0) {
    return 'outHandle';
  }

  const getDirectionScore = (target: Coordinate) => {
    const direction = {
      x: target.x - point.anchor.x,
      y: target.y - point.anchor.y,
    };
    const directionLength = Math.hypot(direction.x, direction.y);
    if (directionLength === 0) {
      return -Infinity;
    }

    return (dragVector.x * direction.x + dragVector.y * direction.y) / directionLength;
  };

  return getDirectionScore(previousPoint.anchor) > getDirectionScore(nextPoint.anchor) ? 'inHandle' : 'outHandle';
};

const getSegmentData = (from: RenderableBezierPoint, to: RenderableBezierPoint): string => {
  const controlPoint1 = from.outHandle ?? from.anchor;
  const controlPoint2 = to.inHandle ?? to.anchor;
  const isLinearSegment =
    controlPoint1.x === from.anchor.x &&
    controlPoint1.y === from.anchor.y &&
    controlPoint2.x === to.anchor.x &&
    controlPoint2.y === to.anchor.y;

  if (isLinearSegment) {
    return `L ${formatCoordinate(to.anchor)}`;
  }

  return `C ${formatCoordinate(controlPoint1)} ${formatCoordinate(controlPoint2)} ${formatCoordinate(to.anchor)}`;
};

export const buildBezierPathData = (points: RenderableBezierPoint[], isClosed: boolean): string => {
  const firstPoint = points[0];
  if (!firstPoint) {
    return '';
  }

  const commands = [`M ${formatCoordinate(firstPoint.anchor)}`];

  for (let i = 1; i < points.length; i += 1) {
    const previousPoint = points[i - 1];
    const currentPoint = points[i];
    if (!previousPoint || !currentPoint) {
      continue;
    }
    commands.push(getSegmentData(previousPoint, currentPoint));
  }

  if (isClosed && points.length > 1) {
    const lastPoint = points.at(-1);
    if (lastPoint) {
      commands.push(getSegmentData(lastPoint, firstPoint));
      commands.push('Z');
    }
  }

  return commands.join(' ');
};

export const anchorsToBezierPoints = (anchors: Coordinate[]): CanvasBezierPointState[] => {
  return anchors.map((anchor) => ({
    anchor,
    inHandle: null,
    outHandle: null,
    type: 'corner',
  }));
};

const evaluateCubicBezier = (segment: CubicBezierSegment, t: number): Coordinate => {
  const [p0, p1, p2, p3] = segment;
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;

  return {
    x: mt2 * mt * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t2 * t * p3.x,
    y: mt2 * mt * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t2 * t * p3.y,
  };
};

const evaluateCubicBezierDerivative = (segment: CubicBezierSegment, t: number): Coordinate => {
  const [p0, p1, p2, p3] = segment;
  const mt = 1 - t;

  return scaleCoordinate(
    addCoordinate(
      addCoordinate(
        scaleCoordinate(subtractCoordinate(p1, p0), mt * mt),
        scaleCoordinate(subtractCoordinate(p2, p1), 2 * mt * t)
      ),
      scaleCoordinate(subtractCoordinate(p3, p2), t * t)
    ),
    3
  );
};

const evaluateCubicBezierSecondDerivative = (segment: CubicBezierSegment, t: number): Coordinate => {
  const [p0, p1, p2, p3] = segment;
  const first = addCoordinate(subtractCoordinate(p2, scaleCoordinate(p1, 2)), p0);
  const second = addCoordinate(subtractCoordinate(p3, scaleCoordinate(p2, 2)), p1);
  return scaleCoordinate(addCoordinate(scaleCoordinate(first, 1 - t), scaleCoordinate(second, t)), 6);
};

const chordLengthParameterize = (points: Coordinate[], first: number, last: number): number[] => {
  const parameters = [0];
  for (let i = first + 1; i <= last; i += 1) {
    const previous = points[i - 1];
    const current = points[i];
    if (!previous || !current) {
      continue;
    }
    parameters.push((parameters.at(-1) ?? 0) + Math.sqrt(getSquaredDistance(previous, current)));
  }

  const totalLength = parameters.at(-1) ?? 0;
  if (totalLength <= BEZIER_FIT_EPSILON) {
    return parameters.map((_, index) => index / Math.max(1, parameters.length - 1));
  }
  return parameters.map((parameter) => parameter / totalLength);
};

const generateBezierSegment = (
  points: Coordinate[],
  first: number,
  last: number,
  parameters: number[],
  leftTangent: Coordinate,
  rightTangent: Coordinate
): CubicBezierSegment => {
  const start = points[first];
  const end = points[last];
  if (!start || !end) {
    throw new Error('Cannot fit a Bezier segment without endpoints');
  }

  let c00 = 0;
  let c01 = 0;
  let c11 = 0;
  let x0 = 0;
  let x1 = 0;

  for (let i = 0; i <= last - first; i += 1) {
    const point = points[first + i];
    const parameter = parameters[i];
    if (!point || parameter === undefined) {
      continue;
    }

    const mt = 1 - parameter;
    const b0 = mt ** 3;
    const b1 = 3 * parameter * mt ** 2;
    const b2 = 3 * parameter ** 2 * mt;
    const b3 = parameter ** 3;
    const a1 = scaleCoordinate(leftTangent, b1);
    const a2 = scaleCoordinate(rightTangent, b2);
    const basePoint = addCoordinate(scaleCoordinate(start, b0 + b1), scaleCoordinate(end, b2 + b3));
    const residual = subtractCoordinate(point, basePoint);

    c00 += dotCoordinates(a1, a1);
    c01 += dotCoordinates(a1, a2);
    c11 += dotCoordinates(a2, a2);
    x0 += dotCoordinates(a1, residual);
    x1 += dotCoordinates(a2, residual);
  }

  const determinant = c00 * c11 - c01 * c01;
  let alphaLeft = determinant === 0 ? 0 : (x0 * c11 - x1 * c01) / determinant;
  let alphaRight = determinant === 0 ? 0 : (c00 * x1 - c01 * x0) / determinant;
  const segmentLength = Math.sqrt(getSquaredDistance(start, end));
  const minimumAlpha = segmentLength * 1e-6;

  if (alphaLeft < minimumAlpha || alphaRight < minimumAlpha) {
    alphaLeft = segmentLength / 3;
    alphaRight = segmentLength / 3;
  }

  return [
    start,
    addCoordinate(start, scaleCoordinate(leftTangent, alphaLeft)),
    addCoordinate(end, scaleCoordinate(rightTangent, alphaRight)),
    end,
  ];
};

const getMaximumBezierFitError = (
  points: Coordinate[],
  first: number,
  last: number,
  segment: CubicBezierSegment,
  parameters: number[]
): { error: number; splitPoint: number } => {
  let maximumError = 0;
  let splitPoint = Math.floor((first + last) / 2);

  for (let i = first + 1; i < last; i += 1) {
    const point = points[i];
    const parameter = parameters[i - first];
    if (!point || parameter === undefined) {
      continue;
    }
    const error = getSquaredDistance(evaluateCubicBezier(segment, parameter), point);
    if (error > maximumError) {
      maximumError = error;
      splitPoint = i;
    }
  }

  return { error: maximumError, splitPoint };
};

const reparameterizeBezierFit = (
  points: Coordinate[],
  first: number,
  parameters: number[],
  segment: CubicBezierSegment
): number[] => {
  return parameters.map((parameter, index) => {
    const point = points[first + index];
    if (!point) {
      return parameter;
    }

    const curvePoint = evaluateCubicBezier(segment, parameter);
    const firstDerivative = evaluateCubicBezierDerivative(segment, parameter);
    const secondDerivative = evaluateCubicBezierSecondDerivative(segment, parameter);
    const difference = subtractCoordinate(curvePoint, point);
    const denominator = dotCoordinates(firstDerivative, firstDerivative) + dotCoordinates(difference, secondDerivative);
    if (Math.abs(denominator) <= BEZIER_FIT_EPSILON) {
      return parameter;
    }
    return Math.max(0, Math.min(1, parameter - dotCoordinates(difference, firstDerivative) / denominator));
  });
};

const fitCubicBezierSegments = (
  points: Coordinate[],
  first: number,
  last: number,
  leftTangent: Coordinate,
  rightTangent: Coordinate,
  maximumErrorSquared: number,
  segments: CubicBezierSegment[]
) => {
  const start = points[first];
  const end = points[last];
  if (!start || !end) {
    return;
  }

  if (last - first === 1) {
    const handleLength = Math.sqrt(getSquaredDistance(start, end)) / 3;
    segments.push([
      start,
      addCoordinate(start, scaleCoordinate(leftTangent, handleLength)),
      addCoordinate(end, scaleCoordinate(rightTangent, handleLength)),
      end,
    ]);
    return;
  }

  let parameters = chordLengthParameterize(points, first, last);
  let segment = generateBezierSegment(points, first, last, parameters, leftTangent, rightTangent);
  let fit = getMaximumBezierFitError(points, first, last, segment, parameters);

  if (fit.error <= maximumErrorSquared) {
    segments.push(segment);
    return;
  }

  if (fit.error <= maximumErrorSquared * 4) {
    for (let i = 0; i < 4; i += 1) {
      parameters = reparameterizeBezierFit(points, first, parameters, segment);
      segment = generateBezierSegment(points, first, last, parameters, leftTangent, rightTangent);
      fit = getMaximumBezierFitError(points, first, last, segment, parameters);
      if (fit.error <= maximumErrorSquared) {
        segments.push(segment);
        return;
      }
    }
  }

  const splitPoint = fit.splitPoint;
  const beforeSplit = points[splitPoint - 1];
  const afterSplit = points[splitPoint + 1];
  if (!beforeSplit || !afterSplit) {
    return;
  }
  const centerTangent = normalizeCoordinate(subtractCoordinate(beforeSplit, afterSplit));

  fitCubicBezierSegments(points, first, splitPoint, leftTangent, centerTangent, maximumErrorSquared, segments);
  fitCubicBezierSegments(
    points,
    splitPoint,
    last,
    scaleCoordinate(centerTangent, -1),
    rightTangent,
    maximumErrorSquared,
    segments
  );
};

export const fitPolylineToBezierPoints = (
  inputPoints: Coordinate[],
  maximumError: number
): CanvasBezierPointState[] => {
  const points = inputPoints.filter((point, index) => {
    const previous = inputPoints[index - 1];
    return !previous || getSquaredDistance(point, previous) > BEZIER_FIT_EPSILON;
  });
  const firstPoint = points[0];
  const secondPoint = points[1];
  const lastPoint = points.at(-1);
  const penultimatePoint = points.at(-2);
  if (!firstPoint || !secondPoint || !lastPoint || !penultimatePoint) {
    return [];
  }

  const segments: CubicBezierSegment[] = [];
  fitCubicBezierSegments(
    points,
    0,
    points.length - 1,
    normalizeCoordinate(subtractCoordinate(secondPoint, firstPoint)),
    normalizeCoordinate(subtractCoordinate(penultimatePoint, lastPoint)),
    Math.max(maximumError, BEZIER_FIT_EPSILON) ** 2,
    segments
  );
  const firstSegment = segments[0];
  if (!firstSegment) {
    return [];
  }

  const bezierPoints: CanvasBezierPointState[] = [
    {
      anchor: firstSegment[0],
      inHandle: null,
      outHandle: normalizeHandle(firstSegment[0], firstSegment[1]),
      type: 'smooth',
    },
  ];

  for (const segment of segments) {
    const previousPoint = bezierPoints.at(-1);
    if (!previousPoint) {
      continue;
    }
    previousPoint.outHandle = normalizeHandle(previousPoint.anchor, segment[1]);
    bezierPoints.push({
      anchor: segment[3],
      inHandle: normalizeHandle(segment[3], segment[2]),
      outHandle: null,
      type: 'smooth',
    });
  }

  return bezierPoints;
};

export const rectToBezierPoints = (rect: Rect): CanvasBezierPointState[] => {
  const { x, y, width, height } = rect;
  return anchorsToBezierPoints([
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ]);
};

export const ovalToBezierPoints = (rect: Rect): CanvasBezierPointState[] => {
  const radiusX = rect.width / 2;
  const radiusY = rect.height / 2;
  const centerX = rect.x + radiusX;
  const centerY = rect.y + radiusY;
  const controlOffsetX = radiusX * ELLIPSE_BEZIER_CONTROL_POINT_RATIO;
  const controlOffsetY = radiusY * ELLIPSE_BEZIER_CONTROL_POINT_RATIO;

  return [
    {
      anchor: { x: centerX, y: rect.y },
      inHandle: { x: centerX - controlOffsetX, y: rect.y },
      outHandle: { x: centerX + controlOffsetX, y: rect.y },
      type: 'symmetric',
    },
    {
      anchor: { x: rect.x + rect.width, y: centerY },
      inHandle: { x: rect.x + rect.width, y: centerY - controlOffsetY },
      outHandle: { x: rect.x + rect.width, y: centerY + controlOffsetY },
      type: 'symmetric',
    },
    {
      anchor: { x: centerX, y: rect.y + rect.height },
      inHandle: { x: centerX + controlOffsetX, y: rect.y + rect.height },
      outHandle: { x: centerX - controlOffsetX, y: rect.y + rect.height },
      type: 'symmetric',
    },
    {
      anchor: { x: rect.x, y: centerY },
      inHandle: { x: rect.x, y: centerY + controlOffsetY },
      outHandle: { x: rect.x, y: centerY - controlOffsetY },
      type: 'symmetric',
    },
  ];
};

const getSmoothTangent = (
  previousAnchor: Coordinate | null,
  anchor: Coordinate,
  nextAnchor: Coordinate | null
): Coordinate => {
  const previousDistance = previousAnchor ? getDistance(previousAnchor, anchor) : 0;
  const nextDistance = nextAnchor ? getDistance(anchor, nextAnchor) : 0;
  const previousInterval = previousDistance ** CENTRIPETAL_CATMULL_ROM_ALPHA;
  const nextInterval = nextDistance ** CENTRIPETAL_CATMULL_ROM_ALPHA;

  if (previousAnchor && nextAnchor && previousInterval > BEZIER_FIT_EPSILON && nextInterval > BEZIER_FIT_EPSILON) {
    const tangent = addCoordinate(
      subtractCoordinate(
        scaleCoordinate(subtractCoordinate(anchor, previousAnchor), 1 / previousInterval),
        scaleCoordinate(subtractCoordinate(nextAnchor, previousAnchor), 1 / (previousInterval + nextInterval))
      ),
      scaleCoordinate(subtractCoordinate(nextAnchor, anchor), 1 / nextInterval)
    );
    return tangent;
  }

  if (nextAnchor && nextInterval > BEZIER_FIT_EPSILON) {
    return scaleCoordinate(subtractCoordinate(nextAnchor, anchor), 1 / nextInterval);
  }

  if (previousAnchor && previousInterval > BEZIER_FIT_EPSILON) {
    return scaleCoordinate(subtractCoordinate(anchor, previousAnchor), 1 / previousInterval);
  }

  return { x: 0, y: 0 };
};

const getSmoothTurningAngle = (
  previousAnchor: Coordinate | null,
  anchor: Coordinate,
  nextAnchor: Coordinate | null
): number => {
  if (!previousAnchor || !nextAnchor) {
    return 0;
  }

  const incomingDirection = normalizeCoordinate(subtractCoordinate(anchor, previousAnchor));
  const outgoingDirection = normalizeCoordinate(subtractCoordinate(nextAnchor, anchor));
  const directionDot = Math.max(-1, Math.min(1, dotCoordinates(incomingDirection, outgoingDirection)));
  return Math.acos(directionDot);
};

const getCircularArcHandleLength = (segmentLength: number, turningAngle: number): number => {
  const cosine = Math.cos(turningAngle / 4);
  const circularArcLength = segmentLength / (3 * cosine * cosine);
  return Math.min(circularArcLength, segmentLength * SMOOTH_HANDLE_MAX_SEGMENT_RATIO);
};

export const smoothBezierPathPoints = (
  points: CanvasBezierPointState[],
  isClosed: boolean
): CanvasBezierPointState[] => {
  if (points.length < 2) {
    return points.map((point) => ({ ...point }));
  }

  const lastPointIndex = points.length - 1;

  return points.map((point, pointIndex) => {
    if (point.type === 'symmetric' && point.inHandle && point.outHandle) {
      return { ...point };
    }

    const previousPoint = isClosed ? points[(pointIndex - 1 + points.length) % points.length] : points[pointIndex - 1];
    const nextPoint = isClosed ? points[(pointIndex + 1) % points.length] : points[pointIndex + 1];
    const previousAnchor = previousPoint?.anchor ?? null;
    const nextAnchor = nextPoint?.anchor ?? null;
    const tangentDirection = normalizeCoordinate(getSmoothTangent(previousAnchor, point.anchor, nextAnchor));
    const turningAngle = getSmoothTurningAngle(previousAnchor, point.anchor, nextAnchor);
    const inHandleLength = getCircularArcHandleLength(
      previousAnchor ? getDistance(point.anchor, previousAnchor) : 0,
      turningAngle
    );
    const outHandleLength = getCircularArcHandleLength(
      nextAnchor ? getDistance(point.anchor, nextAnchor) : 0,
      turningAngle
    );

    return {
      ...point,
      inHandle:
        !isClosed && pointIndex === 0
          ? null
          : normalizeHandle(point.anchor, {
              x: point.anchor.x - tangentDirection.x * inHandleLength,
              y: point.anchor.y - tangentDirection.y * inHandleLength,
            }),
      outHandle:
        !isClosed && pointIndex === lastPointIndex
          ? null
          : normalizeHandle(point.anchor, {
              x: point.anchor.x + tangentDirection.x * outHandleLength,
              y: point.anchor.y + tangentDirection.y * outHandleLength,
            }),
      type: 'smooth',
    };
  });
};

export const evaluateBezierSegment = (
  from: RenderableBezierPoint,
  to: RenderableBezierPoint,
  t: number
): Coordinate => {
  const p0 = from.anchor;
  const p1 = from.outHandle ?? from.anchor;
  const p2 = to.inHandle ?? to.anchor;
  const p3 = to.anchor;
  const mt = 1 - t;
  const mt2 = mt * mt;
  const t2 = t * t;

  return {
    x: mt2 * mt * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t2 * t * p3.x,
    y: mt2 * mt * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t2 * t * p3.y,
  };
};

export const splitBezierSegmentAt = (
  from: CanvasBezierPointState,
  to: CanvasBezierPointState,
  t: number
): {
  fromOutHandle: Coordinate | null;
  insertPoint: CanvasBezierPointState;
  toInHandle: Coordinate | null;
} => {
  const p0 = from.anchor;
  const p1 = from.outHandle ?? from.anchor;
  const p2 = to.inHandle ?? to.anchor;
  const p3 = to.anchor;

  const q0 = lerpCoordinate(p0, p1, t);
  const q1 = lerpCoordinate(p1, p2, t);
  const q2 = lerpCoordinate(p2, p3, t);
  const r0 = lerpCoordinate(q0, q1, t);
  const r1 = lerpCoordinate(q1, q2, t);
  const s = lerpCoordinate(r0, r1, t);

  return {
    fromOutHandle: normalizeHandle(from.anchor, q0),
    insertPoint: {
      anchor: s,
      inHandle: normalizeHandle(s, r0),
      outHandle: normalizeHandle(s, r1),
      type: 'smooth',
    },
    toInHandle: normalizeHandle(to.anchor, q2),
  };
};

export const findNearestBezierPathSegment = (
  points: RenderableBezierPoint[],
  isClosed: boolean,
  point: Coordinate,
  samplesPerSegment = DEFAULT_BEZIER_PATH_SAMPLES_PER_SEGMENT
): BezierPathSegmentHit | null => {
  if (points.length < 2) {
    return null;
  }

  let nearestHit: BezierPathSegmentHit | null = null;
  const segmentCount = isClosed ? points.length : points.length - 1;

  for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
    const from = points[segmentIndex];
    const to = points[(segmentIndex + 1) % points.length];
    if (!from || !to) {
      continue;
    }

    let previous = from.anchor;
    for (let sampleIndex = 1; sampleIndex <= samplesPerSegment; sampleIndex += 1) {
      const t = sampleIndex / samplesPerSegment;
      const current = evaluateBezierSegment(from, to, t);
      const segmentVector = {
        x: current.x - previous.x,
        y: current.y - previous.y,
      };
      const segmentLengthSquared = segmentVector.x ** 2 + segmentVector.y ** 2;
      const projectedT =
        segmentLengthSquared === 0
          ? 0
          : ((point.x - previous.x) * segmentVector.x + (point.y - previous.y) * segmentVector.y) /
            segmentLengthSquared;
      const clampedProjectedT = Math.max(0, Math.min(1, projectedT));
      const nearestPoint = {
        x: previous.x + segmentVector.x * clampedProjectedT,
        y: previous.y + segmentVector.y * clampedProjectedT,
      };
      const distance = getDistance(point, nearestPoint);

      if (!nearestHit || distance < nearestHit.distance) {
        nearestHit = {
          segmentIndex,
          t: (sampleIndex - 1 + clampedProjectedT) / samplesPerSegment,
          point: nearestPoint,
          distance,
        };
      }

      previous = current;
    }
  }

  return nearestHit;
};

export const getBezierPathHitSamplesPerSegment = (stageScale: number): number => {
  const normalizedScale = Math.max(1, stageScale);
  return Math.ceil(DEFAULT_BEZIER_PATH_SAMPLES_PER_SEGMENT * Math.sqrt(normalizedScale));
};

export const approximateBezierPath = (
  points: RenderableBezierPoint[],
  isClosed: boolean,
  samplesPerSegment = DEFAULT_BEZIER_PATH_SAMPLES_PER_SEGMENT
): Coordinate[] => {
  const firstPoint = points[0];
  if (!firstPoint) {
    return [];
  }

  if (points.length === 1) {
    return [firstPoint.anchor];
  }

  const approximatedPoints: Coordinate[] = [firstPoint.anchor];
  const segmentCount = isClosed ? points.length : points.length - 1;

  for (let segmentIndex = 0; segmentIndex < segmentCount; segmentIndex += 1) {
    const from = points[segmentIndex];
    const to = points[(segmentIndex + 1) % points.length];
    if (!from || !to) {
      continue;
    }

    for (let sampleIndex = 1; sampleIndex <= samplesPerSegment; sampleIndex += 1) {
      approximatedPoints.push(evaluateBezierSegment(from, to, sampleIndex / samplesPerSegment));
    }
  }

  return approximatedPoints;
};
