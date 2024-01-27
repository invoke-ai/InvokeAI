import { Badge, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { memo, useMemo } from 'react';
import type { EdgeProps } from 'reactflow';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from 'reactflow';

import { makeEdgeSelector } from './util/makeEdgeSelector';

const InvocationCollapsedEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
  selected,
  source,
  target,
  sourceHandleId,
  targetHandleId,
}: EdgeProps<{ count: number }>) => {
  const selector = useMemo(
    () => makeEdgeSelector(source, sourceHandleId, target, targetHandleId, selected),
    [selected, source, sourceHandleId, target, targetHandleId]
  );

  const { isSelected, shouldAnimate } = useAppSelector(selector);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const { base500 } = useChakraThemeTokens();

  const edgeStyles = useMemo(
    () => ({
      strokeWidth: isSelected ? 3 : 2,
      stroke: base500,
      opacity: isSelected ? 0.8 : 0.5,
      animation: shouldAnimate ? 'dashdraw 0.5s linear infinite' : undefined,
      strokeDasharray: shouldAnimate ? 5 : 'none',
    }),
    [base500, isSelected, shouldAnimate]
  );

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={edgeStyles} />
      {data?.count && data.count > 1 && (
        <EdgeLabelRenderer>
          <Flex
            position="absolute"
            transform={`translate(-50%, -50%) translate(${labelX}px,${labelY}px)`}
            className="nodrag nopan"
          >
            <Badge variant="solid" bg="base.500" opacity={isSelected ? 0.8 : 0.5} boxShadow="base">
              {data.count}
            </Badge>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationCollapsedEdge);
