import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';

import { getFieldColor } from './getEdgeColor';

export const makeEdgeSelector = (
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined,
  selected?: boolean
) =>
  createMemoizedSelector(selectNodesSlice, (nodes) => {
    const sourceNode = nodes.nodes.find((node) => node.id === source);
    const targetNode = nodes.nodes.find((node) => node.id === target);

    const isInvocationToInvocationEdge = isInvocationNode(sourceNode) && isInvocationNode(targetNode);

    const isSelected = sourceNode?.selected || targetNode?.selected || selected;
    const sourceType = isInvocationToInvocationEdge ? sourceNode?.data?.outputs[sourceHandleId || '']?.type : undefined;

    const stroke = sourceType && nodes.shouldColorEdges ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');

    return {
      isSelected,
      shouldAnimate: nodes.shouldAnimateEdges && isSelected,
      stroke,
    };
  });
