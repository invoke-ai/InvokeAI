import { createSelector } from '@reduxjs/toolkit';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { selectNodes } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';

import { getFieldColor } from './getEdgeColor';

export const buildSelectAreConnectedNodesSelected = (source: string, target: string) =>
  createSelector(selectNodes, (nodes): boolean => {
    const sourceNode = nodes.find((node) => node.id === source);
    const targetNode = nodes.find((node) => node.id === target);

    return Boolean(sourceNode?.selected || targetNode?.selected);
  });

export const buildSelectEdgeColor = (
  templates: Templates,
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined
) =>
  createSelector(selectNodes, selectWorkflowSettingsSlice, (nodes, workflowSettings): string => {
    const { shouldColorEdges } = workflowSettings;
    if (!shouldColorEdges) {
      return colorTokenToCssVar('base.500');
    }
    const sourceNode = nodes.find((node) => node.id === source);
    const targetNode = nodes.find((node) => node.id === target);

    if (!sourceNode || !sourceHandleId || !targetNode || !targetHandleId) {
      return colorTokenToCssVar('base.500');
    }

    const sourceNodeTemplate = templates[sourceNode.data.type];

    const isInvocationToInvocationEdge = isInvocationNode(sourceNode) && isInvocationNode(targetNode);
    const outputFieldTemplate = sourceNodeTemplate?.outputs[sourceHandleId];
    const sourceType = isInvocationToInvocationEdge ? outputFieldTemplate?.type : undefined;

    return sourceType ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');
  });

export const buildSelectEdgeLabel = (
  templates: Templates,
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined
) =>
  createSelector(selectNodes, (nodes): string | null => {
    const sourceNode = nodes.find((node) => node.id === source);
    const targetNode = nodes.find((node) => node.id === target);

    if (!sourceNode || !sourceHandleId || !targetNode || !targetHandleId) {
      return null;
    }

    const sourceNodeTemplate = templates[sourceNode.data.type];
    const targetNodeTemplate = templates[targetNode.data.type];

    return `${sourceNodeTemplate?.title || sourceNode.data?.label} -> ${targetNodeTemplate?.title || targetNode.data?.label}`;
  });
