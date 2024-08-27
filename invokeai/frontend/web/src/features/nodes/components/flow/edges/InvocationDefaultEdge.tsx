import { Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { getEdgeStyles } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectShouldShowEdgeLabels } from 'features/nodes/store/workflowSettingsSlice';
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
  selected = false,
  source,
  target,
  sourceHandleId,
  targetHandleId,
}: EdgeProps) => {
  const templates = useStore($templates);
  const selector = useMemo(
    () => makeEdgeSelector(templates, source, sourceHandleId, target, targetHandleId),
    [templates, source, sourceHandleId, target, targetHandleId]
  );

  const { shouldAnimateEdges, areConnectedNodesSelected, stroke, label } = useAppSelector(selector);
  const shouldShowEdgeLabels = useAppSelector(selectShouldShowEdgeLabels);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const edgeStyles = useMemo(
    () => getEdgeStyles(stroke, selected, shouldAnimateEdges, areConnectedNodesSelected),
    [areConnectedNodesSelected, stroke, selected, shouldAnimateEdges]
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
            borderColor={selected ? 'undefined' : 'transparent'}
            opacity={selected ? 1 : 0.5}
            py={1}
            px={3}
            shadow="md"
          >
            <Text size="sm" fontWeight="semibold" color={selected ? 'base.100' : 'base.300'}>
              {label}
            </Text>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationDefaultEdge);
