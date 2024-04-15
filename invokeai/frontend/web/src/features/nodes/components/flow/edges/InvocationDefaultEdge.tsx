import { Flex, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { CSSProperties } from 'react';
import { memo, useMemo } from 'react';
import type { EdgeProps } from 'reactflow';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from 'reactflow';

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

  const { isSelected, shouldAnimate, stroke, label } = useAppSelector(selector);
  const shouldShowEdgeLabels = useAppSelector((s) => s.nodes.shouldShowEdgeLabels);

  const [edgePath, labelX, labelY] = getBezierPath({
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

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={edgeStyles} />
      {label && shouldShowEdgeLabels && (
        <EdgeLabelRenderer>
          <Flex
            className="nodrag nopan"
            pointerEvents="all"
            position="absolute"
            transform={`translate(-50%, -50%) translate(${labelX}px,${labelY}px)`}
            bg="base.800"
            borderRadius="base"
            borderWidth={1}
            borderColor={isSelected ? 'undefined' : 'transparent'}
            opacity={isSelected ? 1 : 0.5}
            py={1}
            px={3}
            shadow="md"
          >
            <Text size="sm" fontWeight="semibold" color={isSelected ? 'base.100' : 'base.300'}>
              {label}
            </Text>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationDefaultEdge);
