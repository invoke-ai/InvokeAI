import { useAppSelector } from 'app/store/storeHooks';
import { memo, useMemo } from 'react';
import { BaseEdge, EdgeProps, getBezierPath } from 'reactflow';
import { makeEdgeSelector } from './util/makeEdgeSelector';

const InvocationDefaultEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  selected,
  source,
  target,
  sourceHandleId,
  targetHandleId,
}: EdgeProps) => {
  const selector = useMemo(
    () =>
      makeEdgeSelector(
        source,
        sourceHandleId,
        target,
        targetHandleId,
        selected
      ),
    [source, sourceHandleId, target, targetHandleId, selected]
  );

  const { isSelected, shouldAnimate, stroke } = useAppSelector(selector);

  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <BaseEdge
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        strokeWidth: isSelected ? 3 : 2,
        stroke,
        opacity: isSelected ? 0.8 : 0.5,
        animation: shouldAnimate ? 'dashdraw 0.5s linear infinite' : undefined,
        strokeDasharray: shouldAnimate ? 5 : 'none',
      }}
    />
  );
};

export default memo(InvocationDefaultEdge);
