import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { chakra, Flex, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import type { EdgeProps } from '@xyflow/react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from '@xyflow/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectShouldAnimateEdges, selectShouldShowEdgeLabels } from 'features/nodes/store/workflowSettingsSlice';
import { NO_DRAG_CLASS, NO_PAN_CLASS } from 'features/nodes/types/constants';
import type { DefaultInvocationNodeEdge } from 'features/nodes/types/invocation';
import { memo, useMemo } from 'react';

import {
  buildSelectAreConnectedNodesSelected,
  buildSelectEdgeColor,
  buildSelectEdgeLabel,
} from './util/buildEdgeSelectors';

const ChakraBaseEdge = chakra(BaseEdge);

const baseEdgeSx: SystemStyleObject = {
  strokeWidth: '3px !important',
  opacity: '0.5 !important',
  strokeDasharray: 'none',
  '&[data-selected="true"]': {
    opacity: '1 !important',
  },
  '&[data-should-animate-edges="true"]': {
    animation: 'dashdraw 0.5s linear infinite !important',
    '&[data-selected="true"], &[data-are-connected-nodes-selected="true"]': {
      strokeDasharray: '5 !important',
    },
  },
};

const edgeLabelWrapperSx: SystemStyleObject = {
  pointerEvents: 'all',
  position: 'absolute',
  bg: 'base.800',
  borderRadius: 'base',
  borderWidth: 1,
  opacity: 0.5,
  borderColor: 'transparent',
  py: 1,
  px: 3,
  shadow: 'md',
  '&[data-selected="true"]': {
    opacity: 1,
    borderColor: undefined,
  },
};

const edgeLabelTextSx: SystemStyleObject = {
  fontWeight: 'semibold',
  color: 'base.300',
  '&[data-selected="true"]': {
    color: 'base.100',
  },
};

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
}: EdgeProps<DefaultInvocationNodeEdge>) => {
  const templates = useStore($templates);
  const shouldAnimateEdges = useAppSelector(selectShouldAnimateEdges);
  const shouldShowEdgeLabels = useAppSelector(selectShouldShowEdgeLabels);

  const selectAreConnectedNodesSelected = useMemo(
    () => buildSelectAreConnectedNodesSelected(source, target),
    [source, target]
  );
  const selectStrokeColor = useMemo(
    () => buildSelectEdgeColor(templates, source, sourceHandleId, target, targetHandleId),
    [templates, source, sourceHandleId, target, targetHandleId]
  );
  const selectEdgeLabel = useMemo(
    () => buildSelectEdgeLabel(templates, source, sourceHandleId, target, targetHandleId),
    [templates, source, sourceHandleId, target, targetHandleId]
  );
  const areConnectedNodesSelected = useAppSelector(selectAreConnectedNodesSelected);
  const stroke = useAppSelector(selectStrokeColor);
  const label = useAppSelector(selectEdgeLabel);

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
        stroke={`${stroke} !important`}
        data-selected={selected}
        data-are-connected-nodes-selected={areConnectedNodesSelected}
        data-should-animate-edges={shouldAnimateEdges}
      />
      {label && shouldShowEdgeLabels && (
        <EdgeLabelRenderer>
          <Flex
            className={`${NO_DRAG_CLASS} ${NO_PAN_CLASS}`}
            transform={`translate(-50%, -50%) translate(${labelX}px,${labelY}px)`}
            data-selected={selected}
            sx={edgeLabelWrapperSx}
          >
            <Text size="sm" sx={edgeLabelTextSx} data-selected={selected}>
              {label}
            </Text>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

export default memo(InvocationDefaultEdge);
