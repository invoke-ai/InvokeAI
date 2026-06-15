import type { FieldType, InvocationTemplates, WorkflowEdge, WorkflowNode } from './types';

import { isConnectorNode, isInvocationNode } from './types';

export const CONNECTOR_INPUT_HANDLE = 'in';
export const CONNECTOR_OUTPUT_HANDLE = 'out';

export interface ResolvedConnectorSource {
  fieldName: string;
  nodeId: string;
  type: FieldType | null;
}

export interface ResolvedWorkflowEdge extends WorkflowEdge {
  source: string;
  sourceHandle: string;
}

export const getConnectorInputEdge = (connectorId: string, edges: WorkflowEdge[]): WorkflowEdge | undefined =>
  edges.find(
    (edge) => edge.type === 'default' && edge.target === connectorId && edge.targetHandle === CONNECTOR_INPUT_HANDLE
  );

export const getConnectorOutputEdges = (connectorId: string, edges: WorkflowEdge[]): WorkflowEdge[] =>
  edges.filter(
    (edge) => edge.type === 'default' && edge.source === connectorId && edge.sourceHandle === CONNECTOR_OUTPUT_HANDLE
  );

export const resolveConnectorSource = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorSource | null => {
  const visited = new Set<string>();

  const resolve = (nodeId: string): ResolvedConnectorSource | null => {
    if (visited.has(nodeId)) {
      return null;
    }

    visited.add(nodeId);

    const inboundEdge = getConnectorInputEdge(nodeId, edges);

    if (!inboundEdge) {
      return null;
    }

    const sourceNode = nodes.find((node) => node.id === inboundEdge.source);

    if (!sourceNode) {
      return null;
    }

    if (isInvocationNode(sourceNode)) {
      return {
        fieldName: inboundEdge.sourceHandle,
        nodeId: sourceNode.id,
        type: templates?.[sourceNode.data.type]?.outputs[inboundEdge.sourceHandle]?.type ?? null,
      };
    }

    if (isConnectorNode(sourceNode) && inboundEdge.sourceHandle === CONNECTOR_OUTPUT_HANDLE) {
      return resolve(sourceNode.id);
    }

    return null;
  };

  return resolve(connectorId);
};

export const resolveWorkflowEdgeSource = (
  edge: WorkflowEdge,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorSource | null => {
  const sourceNode = nodes.find((node) => node.id === edge.source);

  if (!sourceNode) {
    return null;
  }

  if (isInvocationNode(sourceNode)) {
    return {
      fieldName: edge.sourceHandle,
      nodeId: sourceNode.id,
      type: templates?.[sourceNode.data.type]?.outputs[edge.sourceHandle]?.type ?? null,
    };
  }

  if (isConnectorNode(sourceNode) && edge.sourceHandle === CONNECTOR_OUTPUT_HANDLE) {
    return resolveConnectorSource(sourceNode.id, nodes, edges, templates);
  }

  return null;
};

export const getResolvedWorkflowEdges = (
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedWorkflowEdge[] => {
  const resolved: ResolvedWorkflowEdge[] = [];

  for (const edge of edges) {
    const source = resolveWorkflowEdgeSource(edge, nodes, edges, templates);

    if (!source) {
      continue;
    }

    resolved.push({ ...edge, source: source.nodeId, sourceHandle: source.fieldName });
  }

  return resolved;
};
