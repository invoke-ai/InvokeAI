import { Badge, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { useChakraThemeTokens } from 'common/hooks/useChakraThemeTokens';
import { useMemo } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  EdgeProps,
  getBezierPath,
} from 'reactflow';
import { FIELDS, colorTokenToCssVar } from '../types/constants';
import { isInvocationNode } from '../types/types';

const makeEdgeSelector = (
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined,
  selected?: boolean
) =>
  createSelector(stateSelector, ({ nodes }) => {
    const sourceNode = nodes.nodes.find((node) => node.id === source);
    const targetNode = nodes.nodes.find((node) => node.id === target);

    const isInvocationToInvocationEdge =
      isInvocationNode(sourceNode) && isInvocationNode(targetNode);

    const isSelected = sourceNode?.selected || targetNode?.selected || selected;
    const sourceType = isInvocationToInvocationEdge
      ? sourceNode?.data?.outputs[sourceHandleId || '']?.type
      : undefined;

    const stroke =
      sourceType && nodes.shouldColorEdges
        ? colorTokenToCssVar(FIELDS[sourceType].color)
        : colorTokenToCssVar('base.500');

    return {
      isSelected,
      shouldAnimate: nodes.shouldAnimateEdges && isSelected,
      stroke,
    };
  });

const CollapsedEdge = ({
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
    () =>
      makeEdgeSelector(
        source,
        sourceHandleId,
        target,
        targetHandleId,
        selected
      ),
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

  return (
    <>
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          strokeWidth: isSelected ? 3 : 2,
          stroke: base500,
          opacity: isSelected ? 0.8 : 0.5,
          animation: shouldAnimate
            ? 'dashdraw 0.5s linear infinite'
            : undefined,
          strokeDasharray: shouldAnimate ? 5 : 'none',
        }}
      />
      {data?.count && data.count > 1 && (
        <EdgeLabelRenderer>
          <Flex
            sx={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            }}
            className="nodrag nopan"
          >
            <Badge
              variant="solid"
              sx={{
                bg: 'base.500',
                opacity: isSelected ? 0.8 : 0.5,
                boxShadow: 'base',
              }}
            >
              {data.count}
            </Badge>
          </Flex>
        </EdgeLabelRenderer>
      )}
    </>
  );
};

const DefaultEdge = ({
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

export const edgeTypes = {
  collapsed: CollapsedEdge,
  default: DefaultEdge,
};
