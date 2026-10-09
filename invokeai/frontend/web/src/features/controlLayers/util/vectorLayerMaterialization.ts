import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasBezierPathState,
  CanvasLassoState,
  CanvasPolygonState,
  RgbaColor,
} from 'features/controlLayers/store/types';
import { approximateBezierPath } from 'features/controlLayers/util/bezierPath';

export const isFillableBezierPath = (path: CanvasBezierPathState): boolean => path.isClosed && path.points.length >= 3;

const getClosedPathPoints = (path: CanvasBezierPathState): number[] =>
  approximateBezierPath(path.points, true).flatMap((point) => [point.x, point.y]);

export const buildClosedPathPolygonObjects = (paths: CanvasBezierPathState[], color: RgbaColor): CanvasPolygonState[] =>
  paths.flatMap((path) => {
    if (!isFillableBezierPath(path)) {
      return [];
    }

    return [
      {
        id: getPrefixedId('polygon'),
        type: 'polygon',
        points: getClosedPathPoints(path),
        color,
        compositeOperation: 'source-over',
      },
    ];
  });

export const buildClosedPathLassoObjects = (paths: CanvasBezierPathState[]): CanvasLassoState[] =>
  paths.flatMap((path) => {
    if (!isFillableBezierPath(path)) {
      return [];
    }

    return [
      {
        id: getPrefixedId('lasso'),
        type: 'lasso',
        points: getClosedPathPoints(path),
        compositeOperation: 'source-over',
      },
    ];
  });
