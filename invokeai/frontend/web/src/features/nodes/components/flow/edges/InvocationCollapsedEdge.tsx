import { Badge, Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { getEdgeStyles } from 'features/nodes/components/flow/edges/util/getEdgeColor';
import { makeEdgeSelector } from 'features/nodes/components/flow/edges/util/makeEdgeSelector';
import { $templates } from 'features/nodes/store/nodesSlice';
import { memo, useMemo } from 'react';
import type { EdgeProps } from 'reactflow';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from 'reactflow';

const InvocationCollapsedEdge = ({
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
  selected = false,
  source,
  sourceHandleId,
  target,
  targetHandleId,
}: EdgeProps<{ count: number }>) => {
  const templates = useStore($templates);
  const selector = useMemo(
    () => makeEdgeSelector(templates, source, sourceHandleId, target, targetHandleId),
    [templates, source, sourceHandleId, target, targetHandleId]
  );

  const { shouldAnimateEdges, areConnectedNodesSelected } = useAppSelector(selector);

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
    () => getEdgeStyles(base500, selected, shouldAnimateEdges, areConnectedNodesSelected),
    [areConnectedNodesSelected, base500, selected, shouldAnimateEdges]
  );

  return (
    <>
      <BaseEdge path={edgePath} markerEnd={markerEnd} style={edgeStyles} />
      {data?.count && data.count > 1 && (
        <EdgeLabelRenderer>
          <Flex
            data-testid="asdfasdfasdf"
            position="absolute"
            transform={`translate(-50%, -50%) translate(${labelX}px,${labelY}px)`}
            className="nodrag nopan"
            // Unfortunately edge labels do not get the same zIndex treatment as edges do, so we need to manage this ourselves
            // See: https://github.com/xyflow/xyflow/issues/3658
            zIndex={1001}
          >
            <Badge variant="solid" bg="base.500" opacity={selected ? 0.8 : 0.5} boxShadow="base">
              {data.count}
            </Badge>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationCollapsedEdge);
