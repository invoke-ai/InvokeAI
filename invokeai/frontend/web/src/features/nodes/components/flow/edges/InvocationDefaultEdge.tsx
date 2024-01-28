import { useAppSelector } from 'app/store/storeHooks';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import type { EdgeProps } from 'reactflow';
import { BaseEdge, getBezierPath } from 'reactflow';

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
    () => makeEdgeSelector(source, sourceHandleId, target, targetHandleId, selected),
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

  const edgeStyles = useMemo<CSSProperties>(
    () => ({
      strokeWidth: isSelected ? 3 : 2,
      stroke,
      opacity: isSelected ? 0.8 : 0.5,
      animation: shouldAnimate ? 'dashdraw 0.5s linear infinite' : undefined,
      strokeDasharray: shouldAnimate ? 5 : 'none',
    }),
    [isSelected, shouldAnimate, stroke]
  );

  return <BaseEdge path={edgePath} markerEnd={markerEnd} style={edgeStyles} />;
};

export default memo(InvocationDefaultEdge);
