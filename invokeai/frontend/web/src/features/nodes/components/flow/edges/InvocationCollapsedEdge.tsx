import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Badge, Box, chakra } from '@invoke-ai/ui-library';
import type { EdgeProps } from '@xyflow/react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { buildSelectAreConnectedNodesSelected } from 'features/nodes/components/flow/edges/util/buildEdgeSelectors';
import { selectShouldAnimateEdges } from 'features/nodes/store/workflowSettingsSlice';
import { NO_DRAG_CLASS, NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { CollapsedInvocationNodeEdge } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';

const ChakraBaseEdge = chakra(BaseEdge);

const baseEdgeSx: SystemStyleObject = {
  strokeWidth: '3px !important',
  stroke: 'base.500 !important',
  opacity: '0.5 !important',
  strokeDasharray: 'none',
  '&[data-selected="true"]': {
    opacity: '1 !important',
  },
  '&[data-selected="true"], &[data-are-connected-nodes-selected="true"]': {
    strokeDasharray: '5 !important',
  },
  '&[data-should-animate-edges="true"]': {
    animation: 'dashdraw 0.5s linear infinite !important',
  },
};

const badgeSx: SystemStyleObject = {
  bg: 'base.500',
  opacity: 0.5,
  shadow: 'base',
  '&[data-selected="true"]': {
    opacity: 1,
  },
};

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
  target,
}: EdgeProps<CollapsedInvocationNodeEdge>) => {
  const shouldAnimateEdges = useAppSelector(selectShouldAnimateEdges);
  const selectAreConnectedNodesSelected = useMemo(
    () => buildSelectAreConnectedNodesSelected(source, target),
    [source, target]
  );

  const areConnectedNodesSelected = useAppSelector(selectAreConnectedNodesSelected);

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <ChakraBaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        sx={baseEdgeSx}
        data-selected={selected}
        data-are-connected-nodes-selected={areConnectedNodesSelected}
        data-should-animate-edges={shouldAnimateEdges}
      />
      {data?.count !== undefined && (
        <EdgeLabelRenderer>
          <Box
            position="absolute"
            transform={`translate(-50%, -50%) translate(${labelX}px,${labelY}px)`}
            className={`edge-label-renderer__custom-edge ${NO_DRAG_CLASS} ${NO_PAN_CLASS}`} // Unfortunately edge labels do not get the same zIndex treatment as edges do, so we need to manage this ourselves
            // See: https://github.com/xyflow/xyflow/issues/3658
            zIndex={1001}
          >
            <Badge variant="solid" sx={badgeSx} data-selected={selected}>
              {data.count}
            </Badge>
          </Box>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationCollapsedEdge);
