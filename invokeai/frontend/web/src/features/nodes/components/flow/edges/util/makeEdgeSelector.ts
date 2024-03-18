import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldOutputTemplate } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';

import { getFieldColor } from './getEdgeColor';

const defaultReturnValue = {
  isSelected: false,
  shouldAnimate: false,
  stroke: colorTokenToCssVar('base.500'),
};

export const makeEdgeSelector = (
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined,
  selected?: boolean
) =>
  createMemoizedSelector(selectNodesSlice, (nodes): { isSelected: boolean; shouldAnimate: boolean; stroke: string } => {
    const sourceNode = nodes.nodes.find((node) => node.id === source);
    const targetNode = nodes.nodes.find((node) => node.id === target);

    const isInvocationToInvocationEdge = isInvocationNode(sourceNode) && isInvocationNode(targetNode);

    const isSelected = Boolean(sourceNode?.selected || targetNode?.selected || selected);
    if (!sourceNode || !sourceHandleId) {
      return defaultReturnValue;
    }

    const outputFieldTemplate = selectFieldOutputTemplate(nodes, sourceNode.id, sourceHandleId);
    const sourceType = isInvocationToInvocationEdge ? outputFieldTemplate?.type : undefined;

    const stroke = sourceType && nodes.shouldColorEdges ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');

    return {
      isSelected,
      shouldAnimate: nodes.shouldAnimateEdges && isSelected,
      stroke,
    };
  });
