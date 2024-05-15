import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectFieldOutputTemplate, selectNodeTemplate } from 'features/nodes/store/selectors';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';

import { getFieldColor } from './getEdgeColor';

const defaultReturnValue = {
  isSelected: false,
  shouldAnimate: false,
  stroke: colorTokenToCssVar('base.500'),
  label: '',
};

export const makeEdgeSelector = (
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined,
  selected?: boolean
) =>
  createMemoizedSelector(
    selectNodesSlice,
    selectWorkflowSettingsSlice,
    (nodes, workflowSettings): { isSelected: boolean; shouldAnimate: boolean; stroke: string; label: string } => {
      const sourceNode = nodes.nodes.find((node) => node.id === source);
      const targetNode = nodes.nodes.find((node) => node.id === target);

      const isInvocationToInvocationEdge = isInvocationNode(sourceNode) && isInvocationNode(targetNode);

      const isSelected = Boolean(sourceNode?.selected || targetNode?.selected || selected);
      if (!sourceNode || !sourceHandleId || !targetNode || !targetHandleId) {
        return defaultReturnValue;
      }

      const outputFieldTemplate = selectFieldOutputTemplate(nodes, sourceNode.id, sourceHandleId);
      const sourceType = isInvocationToInvocationEdge ? outputFieldTemplate?.type : undefined;

      const stroke = sourceType && workflowSettings.shouldColorEdges ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');

      const sourceNodeTemplate = selectNodeTemplate(nodes, sourceNode.id);
      const targetNodeTemplate = selectNodeTemplate(nodes, targetNode.id);

      const label = `${sourceNodeTemplate?.title || sourceNode.data?.label} -> ${targetNodeTemplate?.title || targetNode.data?.label}`;

      return {
        isSelected,
        shouldAnimate: workflowSettings.shouldAnimateEdges && isSelected,
        stroke,
        label,
      };
    }
  );
