import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasBezierPathState,
  CanvasBrushLineState,
  CanvasBrushLineWithPressureState,
  Coordinate,
  RgbaColor,
} from 'features/controlLayers/store/types';
import { approximateBezierPath } from 'features/controlLayers/util/bezierPath';

const TAPER_LENGTH_STROKE_WIDTH_SCALE = 6;

const getDistance = (a: Coordinate, b: Coordinate): number => Math.hypot(a.x - b.x, a.y - b.y);
const smoothstep = (value: number): number => value * value * (3 - 2 * value);

export const buildTaperedTracePoints = (coordinates: Coordinate[], strokeWidth: number): number[] => {
  if (coordinates.length === 0) {
    return [];
  }

  const distancesFromStart = new Array<number>(coordinates.length).fill(0);
  for (let index = 1; index < coordinates.length; index++) {
    const previousPoint = coordinates[index - 1];
    const point = coordinates[index];
    if (!previousPoint || !point) {
      continue;
    }
    distancesFromStart[index] = (distancesFromStart[index - 1] ?? 0) + getDistance(previousPoint, point);
  }

  const totalLength = distancesFromStart.at(-1) ?? 0;
  const taperLength = Math.min(totalLength / 2, Math.max(0, strokeWidth * TAPER_LENGTH_STROKE_WIDTH_SCALE));

  return coordinates.flatMap((coordinate, index) => {
    const distanceFromStart = distancesFromStart[index] ?? 0;
    const distanceFromEnd = totalLength - distanceFromStart;
    const distanceFromNearestEnd = Math.min(distanceFromStart, distanceFromEnd);
    const pressure = taperLength === 0 ? 1 : smoothstep(Math.min(1, distanceFromNearestEnd / taperLength));
    return [coordinate.x, coordinate.y, pressure];
  });
};

export const buildVectorTraceObject = (
  path: CanvasBezierPathState,
  strokeWidth: number,
  color: RgbaColor,
  taperEnds: boolean
): CanvasBrushLineState | CanvasBrushLineWithPressureState | null => {
  const coordinates = approximateBezierPath(path.points, path.isClosed);
  if (coordinates.length < 2) {
    return null;
  }

  if (taperEnds && !path.isClosed) {
    return {
      id: getPrefixedId('brush_line_with_pressure'),
      type: 'brush_line_with_pressure',
      strokeWidth,
      points: buildTaperedTracePoints(coordinates, strokeWidth),
      color,
      pressureAffectsWidth: true,
      pressureAffectsOpacity: false,
      clip: null,
    };
  }

  return {
    id: getPrefixedId('brush_line'),
    type: 'brush_line',
    strokeWidth,
    points: coordinates.flatMap((coordinate) => [coordinate.x, coordinate.y]),
    color,
    clip: null,
  };
};
