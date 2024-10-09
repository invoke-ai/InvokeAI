import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { deepClone } from 'common/util/deepClone';
import { selectNodesSlice } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';

import { getFieldColor } from './getEdgeColor';

const defaultReturnValue = {
  areConnectedNodesSelected: false,
  shouldAnimateEdges: false,
  stroke: colorTokenToCssVar('base.500'),
  label: '',
};

export const makeEdgeSelector = (
  templates: Templates,
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined
) =>
  createMemoizedSelector(
    selectNodesSlice,
    selectWorkflowSettingsSlice,
    (
      nodes,
      workflowSettings
    ): { areConnectedNodesSelected: boolean; shouldAnimateEdges: boolean; stroke: string; label: string } => {
      const { shouldAnimateEdges, shouldColorEdges } = workflowSettings;
      const sourceNode = nodes.nodes.find((node) => node.id === source);
      const targetNode = nodes.nodes.find((node) => node.id === target);

      const returnValue = deepClone(defaultReturnValue);
      returnValue.shouldAnimateEdges = shouldAnimateEdges;

      const isInvocationToInvocationEdge = isInvocationNode(sourceNode) && isInvocationNode(targetNode);

      returnValue.areConnectedNodesSelected = Boolean(sourceNode?.selected || targetNode?.selected);
      if (!sourceNode || !sourceHandleId || !targetNode || !targetHandleId) {
        return returnValue;
      }

      const sourceNodeTemplate = templates[sourceNode.data.type];
      const targetNodeTemplate = templates[targetNode.data.type];

      const outputFieldTemplate = sourceNodeTemplate?.outputs[sourceHandleId];
      const sourceType = isInvocationToInvocationEdge ? outputFieldTemplate?.type : undefined;

      returnValue.stroke = sourceType && shouldColorEdges ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');

      returnValue.label = `${sourceNodeTemplate?.title || sourceNode.data?.label} -> ${targetNodeTemplate?.title || targetNode.data?.label}`;

      return returnValue;
    }
  );
