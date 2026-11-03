import { createSelector } from '@reduxjs/toolkit';
import { colorTokenToCssVar } from 'common/util/colorTokenToCssVar';
import { selectEdges, selectNodes } from 'features/nodes/store/selectors';
import type { Templates } from 'features/nodes/store/types';
import { resolveConnectorSource, resolveConnectorSourceFieldType } from 'features/nodes/store/util/connectorTopology';
import { selectWorkflowSettingsSlice } from 'features/nodes/store/workflowSettingsSlice';
import { isConnectorNode } from 'features/nodes/types/invocation';

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
  createSelector(selectNodes, selectEdges, selectWorkflowSettingsSlice, (nodes, edges, workflowSettings): string => {
    const { shouldColorEdges } = workflowSettings;
    if (!shouldColorEdges) {
      return colorTokenToCssVar('base.500');
    }
    const sourceNode = nodes.find((node) => node.id === source);

    if (!sourceNode || !sourceHandleId || !targetHandleId) {
      return colorTokenToCssVar('base.500');
    }

    const sourceType = isConnectorNode(sourceNode)
      ? resolveConnectorSourceFieldType(sourceNode.id, nodes, edges, templates)
      : templates[sourceNode.data.type]?.outputs[sourceHandleId]?.type;

    return sourceType ? getFieldColor(sourceType) : colorTokenToCssVar('base.500');
  });

export const buildSelectEdgeLabel = (
  templates: Templates,
  source: string,
  sourceHandleId: string | null | undefined,
  target: string,
  targetHandleId: string | null | undefined
) =>
  createSelector(selectNodes, selectEdges, (nodes, edges): string | null => {
    const sourceNode = nodes.find((node) => node.id === source);
    const targetNode = nodes.find((node) => node.id === target);

    if (!sourceNode || !sourceHandleId || !targetNode || !targetHandleId) {
      return null;
    }

    const resolvedSource = isConnectorNode(sourceNode) ? resolveConnectorSource(sourceNode.id, nodes, edges) : null;
    const sourceTemplate =
      resolvedSource !== null
        ? templates[nodes.find((node) => node.id === resolvedSource.nodeId)?.data.type ?? '']
        : templates[sourceNode.data.type];
    const targetNodeTemplate = templates[targetNode.data.type];

    return `${sourceTemplate?.title || sourceNode.data?.label} -> ${targetNodeTemplate?.title || targetNode.data?.label}`;
  });
