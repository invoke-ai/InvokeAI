import type { WorkflowEdge, WorkflowNode } from '@features/workflow/contracts';

import { Box, Text } from '@chakra-ui/react';
import { useIsWorkflowImageExport } from '@features/workflow/ui/nodeChrome';
import { getForLoopBodyBoundaries, type LoopBodyBoundaryStatus } from '@features/workflow/utility';
import { type ReactFlowState, useStore, ViewportPortal } from '@xyflow/react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const BOUNDARY_PADDING = 24;

interface BoundaryRect {
  height: number;
  width: number;
  x: number;
  y: number;
}

const getStatusColor = (status: LoopBodyBoundaryStatus) =>
  status === 'complete'
    ? { border: 'border.success', text: 'fg.success' }
    : { border: 'border.warning', text: 'fg.warning' };

/** Bounds of the measured body nodes from xyflow's live lookup, so drags and remeasures update the box. */
const getMeasuredBounds = (nodeLookup: ReactFlowState['nodeLookup'], nodeIds: string[]): BoundaryRect | null => {
  let x = Infinity;
  let y = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;

  for (const nodeId of nodeIds) {
    const node = nodeLookup.get(nodeId);
    const width = node?.measured.width;
    const height = node?.measured.height;

    if (!node || !width || !height) {
      continue;
    }

    const { x: nodeX, y: nodeY } = node.internals.positionAbsolute;

    x = Math.min(x, nodeX);
    y = Math.min(y, nodeY);
    x2 = Math.max(x2, nodeX + width);
    y2 = Math.max(y2, nodeY + height);
  }

  return x2 > x && y2 > y ? { height: y2 - y, width: x2 - x, x, y } : null;
};

const areSameRects = (a: (BoundaryRect | null)[], b: (BoundaryRect | null)[]): boolean =>
  a.length === b.length &&
  a.every((rect, i) => {
    const other = b[i] ?? null;

    return rect === other
      ? true
      : rect !== null &&
          other !== null &&
          rect.x === other.x &&
          rect.y === other.y &&
          rect.width === other.width &&
          rect.height === other.height;
  });

export const LoopBodyBoundaryOverlay = ({ nodes, edges }: { nodes: WorkflowNode[]; edges: WorkflowEdge[] }) => {
  const isWorkflowImageExport = useIsWorkflowImageExport();
  const { t } = useTranslation();
  const boundaries = useMemo(() => getForLoopBodyBoundaries(nodes, edges), [edges, nodes]);
  const selectBounds = useCallback(
    (state: ReactFlowState) => boundaries.map((boundary) => getMeasuredBounds(state.nodeLookup, boundary.bodyNodeIds)),
    [boundaries]
  );
  const boundsByIndex = useStore(selectBounds, areSameRects);

  return (
    <ViewportPortal>
      {boundaries.map((boundary, index) => {
        const bounds = boundsByIndex[index];

        if (!bounds) {
          return null;
        }

        const colors = isWorkflowImageExport
          ? { border: 'border.emphasized', text: 'fg.subtle' }
          : getStatusColor(boundary.status);
        const bodyLabel = t('nodes.forLoopBodyBoundary');
        const statusLabel =
          isWorkflowImageExport || boundary.status === 'complete'
            ? ''
            : t(`nodes.forLoopBodyBoundaryStatus.${boundary.status}`);
        const label = statusLabel ? `${bodyLabel} - ${statusLabel}` : bodyLabel;

        return (
          <Box
            key={`${boundary.forNodeId ?? 'orphan'}-${boundary.returnNodeId ?? 'return'}-${boundary.status}`}
            aria-label={label}
            border="2px dashed"
            borderColor={colors.border}
            borderRadius="base"
            data-loop-body-boundary={boundary.forNodeId ?? boundary.returnNodeId}
            data-loop-body-status={isWorkflowImageExport ? undefined : boundary.status}
            h={bounds.height + BOUNDARY_PADDING * 2}
            pointerEvents="none"
            position="absolute"
            transform={`translate(${bounds.x - BOUNDARY_PADDING}px, ${bounds.y - BOUNDARY_PADDING}px)`}
            w={bounds.width + BOUNDARY_PADDING * 2}
            zIndex={0}
          >
            <Text
              position="absolute"
              top={-6}
              left={8}
              px={1}
              bg="bg.canvas"
              color={colors.text}
              fontSize="md"
              lineHeight="short"
              whiteSpace="nowrap"
            >
              {label}
            </Text>
          </Box>
        );
      })}
    </ViewportPortal>
  );
};

export default memo(LoopBodyBoundaryOverlay);
