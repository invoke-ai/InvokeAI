import type { CanvasBezierPointState, Coordinate } from 'features/controlLayers/store/types';

type RenderableBezierPoint = Pick<CanvasBezierPointState, 'anchor' | 'inHandle' | 'outHandle'>;
type BezierPathSegmentHit = {
  segmentIndex: number;
  t: number;
  point: Coordinate;
  distance: number;
};

const formatCoordinate = (coordinate: Coordinate) => `${coordinate.x} ${coordinate.y}`;
const getDistance = (a: Coordinate, b: Coordinate) => Math.hypot(a.x - b.x, a.y - b.y);
const lerpCoordinate = (a: Coordinate, b: Coordinate, t: number): Coordinate => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
});
const normalizeHandle = (anchor: Coordinate, handle: Coordinate): Coordinate | null =>
  anchor.x === handle.x && anchor.y === handle.y ? null : handle;

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
  samplesPerSegment = 24
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

export const approximateBezierPath = (
  points: RenderableBezierPoint[],
  isClosed: boolean,
  samplesPerSegment = 24
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
