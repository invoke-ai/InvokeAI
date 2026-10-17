import { deepClone } from 'common/util/deepClone';
import type { CanvasBezierPathState, CanvasBezierPointState, Coordinate } from 'features/controlLayers/store/types';
import { setBezierPointType } from 'features/controlLayers/util/bezierPath';

type TopologyResult = {
  path: CanvasBezierPathState;
  activePointIndex: number;
};

type VectorPathPointRef = { pathId: string; pointIndex: number };

type SplitVectorPathResult = {
  paths: CanvasBezierPathState[];
  activePathId: string;
  activePointIndex: number;
};

type DeleteVectorPathPointsResult = {
  paths: CanvasBezierPathState[];
  didDelete: boolean;
};

const translateHandle = (handle: Coordinate | null, from: Coordinate, to: Coordinate): Coordinate | null =>
  handle ? { x: handle.x + to.x - from.x, y: handle.y + to.y - from.y } : null;

const reversePoints = (path: CanvasBezierPathState): CanvasBezierPathState['points'] =>
  [...path.points].reverse().map((point) => ({
    ...point,
    inHandle: point.outHandle,
    outHandle: point.inHandle,
  }));

const prepareJoinedEndpoint = (point: CanvasBezierPointState, connectingHandle: 'inHandle' | 'outHandle') => {
  if (point.type === 'corner') {
    point[connectingHandle] = null;
    return;
  }
  // Keep the existing segment's tangent and restore the handle used by the new segment.
  setBezierPointType(point, point.type, connectingHandle === 'inHandle' ? 'outHandle' : 'inHandle');
};

export const deleteVectorPathPoints = (
  sourcePaths: CanvasBezierPathState[],
  pointRefs: VectorPathPointRef[]
): DeleteVectorPathPointsResult => {
  const selectedPointIndices = new Map<string, Set<number>>();
  for (const pointRef of pointRefs) {
    const indices = selectedPointIndices.get(pointRef.pathId) ?? new Set<number>();
    indices.add(pointRef.pointIndex);
    selectedPointIndices.set(pointRef.pathId, indices);
  }

  let didDelete = false;
  const paths = deepClone(sourcePaths).filter((path) => {
    const pointIndices = [...(selectedPointIndices.get(path.id) ?? [])]
      .filter((pointIndex) => path.points[pointIndex] !== undefined)
      .sort((a, b) => b - a);
    if (pointIndices.length === 0) {
      return true;
    }

    didDelete = true;
    const minPointCount = path.isClosed ? 3 : 2;
    if (path.points.length - pointIndices.length < minPointCount) {
      return false;
    }

    const deletedIndices = new Set(pointIndices);
    path.points = path.points.filter((point, pointIndex, points) => {
      if (deletedIndices.has(pointIndex)) {
        return false;
      }
      const previousIndex = path.isClosed ? (pointIndex - 1 + points.length) % points.length : pointIndex - 1;
      const nextIndex = path.isClosed ? (pointIndex + 1) % points.length : pointIndex + 1;
      const previousDeleted = deletedIndices.has(previousIndex);
      const nextDeleted = deletedIndices.has(nextIndex);
      if (previousDeleted || nextDeleted) {
        // Preserve the tangent on the unchanged side of the surviving point.
        setBezierPointType(point, point.type, nextDeleted && !previousDeleted ? 'inHandle' : 'outHandle');
      }
      return true;
    });
    return true;
  });

  return { paths, didDelete };
};

export const canSplitVectorPathAtPoints = (
  paths: CanvasBezierPathState[],
  pointRefs: VectorPathPointRef[]
): boolean => {
  return pointRefs.some((pointRef) => {
    const path = paths.find((candidate) => candidate.id === pointRef.pathId);
    return Boolean(
      path?.points[pointRef.pointIndex] &&
      (path.isClosed || (pointRef.pointIndex > 0 && pointRef.pointIndex < path.points.length - 1))
    );
  });
};

export const canJoinVectorPathEndpoints = (
  paths: CanvasBezierPathState[],
  pointRefs: VectorPathPointRef[],
  weldRadius: number
): boolean => {
  if (pointRefs.length !== 2) {
    return false;
  }
  const [firstRef, secondRef] = pointRefs;
  if (
    !firstRef ||
    !secondRef ||
    (firstRef.pathId === secondRef.pathId && firstRef.pointIndex === secondRef.pointIndex)
  ) {
    return false;
  }
  const firstPath = paths.find((path) => path.id === firstRef.pathId);
  const secondPath = paths.find((path) => path.id === secondRef.pathId);
  if (!firstPath || !secondPath || firstPath.isClosed || secondPath.isClosed) {
    return false;
  }
  const isEndpoint = (path: CanvasBezierPathState, pointIndex: number) =>
    pointIndex === 0 || pointIndex === path.points.length - 1;
  if (!isEndpoint(firstPath, firstRef.pointIndex) || !isEndpoint(secondPath, secondRef.pointIndex)) {
    return false;
  }
  if (firstPath.id !== secondPath.id) {
    return true;
  }
  const firstPoint = firstPath.points[firstRef.pointIndex];
  const secondPoint = secondPath.points[secondRef.pointIndex];
  if (!firstPoint || !secondPoint) {
    return false;
  }
  const shouldWeld =
    Math.hypot(firstPoint.anchor.x - secondPoint.anchor.x, firstPoint.anchor.y - secondPoint.anchor.y) <= weldRadius;
  return !shouldWeld || firstPath.points.length >= 4;
};

export const splitVectorPathAtPoint = (
  sourcePath: CanvasBezierPathState,
  pointIndex: number,
  newPathId: string
): SplitVectorPathResult | null => splitVectorPathAtPoints(sourcePath, [pointIndex], () => newPathId);

export const splitVectorPathAtPoints = (
  sourcePath: CanvasBezierPathState,
  pointIndices: number[],
  getNewPathId: () => string
): SplitVectorPathResult | null => {
  const path = deepClone(sourcePath);
  const validPointIndices = pointIndices.filter((pointIndex) => path.points[pointIndex] !== undefined);
  const uniquePointIndices = [...new Set(validPointIndices)].sort((a, b) => a - b);
  if (uniquePointIndices.length === 0) {
    return null;
  }

  if (path.isClosed) {
    const splitPaths = uniquePointIndices.map((startIndex, splitIndex) => {
      const endIndex = uniquePointIndices[(splitIndex + 1) % uniquePointIndices.length] ?? startIndex;
      const points: CanvasBezierPathState['points'] = [];
      let pointIndex = startIndex;
      do {
        const point = path.points[pointIndex];
        if (point) {
          points.push(deepClone(point));
        }
        pointIndex = (pointIndex + 1) % path.points.length;
      } while (pointIndex !== endIndex);
      const endPoint = path.points[endIndex];
      if (endPoint) {
        points.push(deepClone(endPoint));
      }
      const firstPoint = points[0];
      const lastPoint = points.at(-1);
      if (firstPoint) {
        firstPoint.inHandle = null;
      }
      if (lastPoint) {
        lastPoint.outHandle = null;
      }
      return {
        ...path,
        id: splitIndex === 0 ? path.id : getNewPathId(),
        isClosed: false,
        points,
      };
    });
    const activeSplitPointIndex = validPointIndices.at(-1);
    const activePath = splitPaths[uniquePointIndices.indexOf(activeSplitPointIndex ?? -1)];
    if (!activePath) {
      return null;
    }
    return { paths: splitPaths, activePathId: activePath.id, activePointIndex: 0 };
  }

  const internalPointIndices = uniquePointIndices.filter(
    (pointIndex) => pointIndex > 0 && pointIndex < path.points.length - 1
  );
  if (internalPointIndices.length === 0) {
    return null;
  }

  const boundaries = [0, ...internalPointIndices, path.points.length - 1];
  const splitPaths = boundaries.slice(0, -1).map((startIndex, splitIndex) => {
    const endIndex = boundaries[splitIndex + 1] ?? startIndex;
    const points = deepClone(path.points.slice(startIndex, endIndex + 1));
    const firstPoint = points[0];
    const lastPoint = points.at(-1);
    if (splitIndex > 0 && firstPoint) {
      firstPoint.inHandle = null;
    }
    if (splitIndex < boundaries.length - 2 && lastPoint) {
      lastPoint.outHandle = null;
    }
    return {
      ...path,
      id: splitIndex === 0 ? path.id : getNewPathId(),
      points,
    };
  });
  const activeInternalPointIndex = [...validPointIndices]
    .reverse()
    .find((pointIndex) => internalPointIndices.includes(pointIndex));
  if (activeInternalPointIndex === undefined) {
    return null;
  }
  const activePath = splitPaths[internalPointIndices.indexOf(activeInternalPointIndex) + 1];
  if (!activePath) {
    return null;
  }
  return { paths: splitPaths, activePathId: activePath.id, activePointIndex: 0 };
};

export const joinVectorPathEndpoints = (
  sourcePathState: CanvasBezierPathState,
  sourcePointIndex: number,
  targetPathState: CanvasBezierPathState,
  targetPointIndex: number,
  shouldWeld: boolean
): TopologyResult | null => {
  if (sourcePathState.isClosed || targetPathState.isClosed) {
    return null;
  }
  const sourcePath = deepClone(sourcePathState);
  const targetPath = sourcePath.id === targetPathState.id ? sourcePath : deepClone(targetPathState);
  const isEndpoint = (path: CanvasBezierPathState, pointIndex: number) =>
    pointIndex === 0 || pointIndex === path.points.length - 1;
  if (!isEndpoint(sourcePath, sourcePointIndex) || !isEndpoint(targetPath, targetPointIndex)) {
    return null;
  }

  const sourcePoint = sourcePath.points[sourcePointIndex];
  const targetPoint = targetPath.points[targetPointIndex];
  if (!sourcePoint || !targetPoint) {
    return null;
  }

  if (sourcePath.id === targetPath.id) {
    const firstPoint = sourcePath.points[0];
    const lastPoint = sourcePath.points.at(-1);
    if (!firstPoint || !lastPoint || (shouldWeld && sourcePath.points.length < 4)) {
      return null;
    }
    if (shouldWeld) {
      const joinedPoint: CanvasBezierPointState = {
        anchor: { ...targetPoint.anchor },
        inHandle: translateHandle(lastPoint.inHandle, lastPoint.anchor, targetPoint.anchor),
        outHandle: translateHandle(firstPoint.outHandle, firstPoint.anchor, targetPoint.anchor),
        type: targetPoint.type,
      };
      setBezierPointType(joinedPoint, joinedPoint.type, targetPointIndex === 0 ? 'outHandle' : 'inHandle');
      sourcePath.points = [joinedPoint, ...sourcePath.points.slice(1, -1)];
      sourcePath.isClosed = true;
      return { path: sourcePath, activePointIndex: 0 };
    }

    prepareJoinedEndpoint(firstPoint, 'inHandle');
    prepareJoinedEndpoint(lastPoint, 'outHandle');
    sourcePath.isClosed = true;
    return { path: sourcePath, activePointIndex: targetPointIndex };
  }

  const sourcePoints = sourcePointIndex === 0 ? reversePoints(sourcePath) : sourcePath.points;
  const targetPoints =
    targetPointIndex === targetPath.points.length - 1 ? reversePoints(targetPath) : targetPath.points;
  const sourceEndPoint = sourcePoints.at(-1);
  const targetStartPoint = targetPoints[0];
  if (!sourceEndPoint || !targetStartPoint) {
    return null;
  }

  if (shouldWeld) {
    const joinedPoint: CanvasBezierPointState = {
      anchor: { ...targetStartPoint.anchor },
      inHandle: translateHandle(sourceEndPoint.inHandle, sourceEndPoint.anchor, targetStartPoint.anchor),
      outHandle: targetStartPoint.outHandle,
      type: targetStartPoint.type,
    };
    setBezierPointType(joinedPoint, joinedPoint.type, 'outHandle');
    return {
      path: {
        ...sourcePath,
        points: [...sourcePoints.slice(0, -1), joinedPoint, ...targetPoints.slice(1)],
        isClosed: false,
      },
      activePointIndex: sourcePoints.length - 1,
    };
  }

  prepareJoinedEndpoint(sourceEndPoint, 'outHandle');
  prepareJoinedEndpoint(targetStartPoint, 'inHandle');
  return {
    path: {
      ...sourcePath,
      points: [...sourcePoints, ...targetPoints],
      isClosed: false,
    },
    activePointIndex: sourcePoints.length,
  };
};
