import type { FieldType, InvocationTemplates, WorkflowEdge, WorkflowNode } from './types';

import { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from './connectorHandles';
import { createWorkflowGraphIndex, type WorkflowGraphIndex } from './graphIndex';
import { isConnectorNode, isInvocationNode } from './types';

export { CONNECTOR_INPUT_HANDLE, CONNECTOR_OUTPUT_HANDLE } from './connectorHandles';

export interface ResolvedConnectorSource {
  fieldName: string;
  nodeId: string;
  type: FieldType | null;
}

export interface ResolvedConnectorTarget {
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

export const getConnectorInputEdgeIndexed = (
  connectorId: string,
  index: WorkflowGraphIndex
): WorkflowEdge | undefined => index.connectorInputById.get(connectorId);

export const getConnectorOutputEdges = (connectorId: string, edges: WorkflowEdge[]): WorkflowEdge[] =>
  edges.filter(
    (edge) => edge.type === 'default' && edge.source === connectorId && edge.sourceHandle === CONNECTOR_OUTPUT_HANDLE
  );

export const getConnectorOutputEdgesIndexed = (connectorId: string, index: WorkflowGraphIndex): WorkflowEdge[] =>
  index.connectorOutputsById.get(connectorId) ?? [];

export const resolveConnectorSource = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorSource | null =>
  resolveConnectorSourceIndexed(connectorId, createWorkflowGraphIndex(nodes, edges), templates);

export const resolveConnectorSourceIndexed = (
  connectorId: string,
  index: WorkflowGraphIndex,
  templates?: InvocationTemplates
): ResolvedConnectorSource | null => resolveConnectorSourceWithCache(connectorId, index, templates, new Map());

const resolveConnectorSourceWithCache = (
  connectorId: string,
  index: WorkflowGraphIndex,
  templates: InvocationTemplates | undefined,
  cache: Map<string, ResolvedConnectorSource | null>
): ResolvedConnectorSource | null => {
  const visited = new Set<string>();

  const resolve = (nodeId: string): ResolvedConnectorSource | null => {
    if (cache.has(nodeId)) {
      return cache.get(nodeId) ?? null;
    }

    if (visited.has(nodeId)) {
      return null;
    }

    visited.add(nodeId);

    const inboundEdge = getConnectorInputEdgeIndexed(nodeId, index);

    if (!inboundEdge) {
      cache.set(nodeId, null);
      return null;
    }

    const sourceNode = index.nodesById.get(inboundEdge.source);

    if (!sourceNode) {
      cache.set(nodeId, null);
      return null;
    }

    if (isInvocationNode(sourceNode)) {
      const source = {
        fieldName: inboundEdge.sourceHandle,
        nodeId: sourceNode.id,
        type: templates?.[sourceNode.data.type]?.outputs[inboundEdge.sourceHandle]?.type ?? null,
      };

      cache.set(nodeId, source);
      return source;
    }

    if (isConnectorNode(sourceNode) && inboundEdge.sourceHandle === CONNECTOR_OUTPUT_HANDLE) {
      const source = resolve(sourceNode.id);

      cache.set(nodeId, source);
      return source;
    }

    cache.set(nodeId, null);
    return null;
  };

  return resolve(connectorId);
};

export const resolveConnectorTarget = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorTarget | null => {
  return resolveConnectorTargets(connectorId, nodes, edges, templates)[0] ?? null;
};

export const resolveConnectorTargets = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorTarget[] =>
  resolveConnectorTargetsIndexed(connectorId, createWorkflowGraphIndex(nodes, edges), templates);

export const resolveConnectorTargetsIndexed = (
  connectorId: string,
  index: WorkflowGraphIndex,
  templates?: InvocationTemplates
): ResolvedConnectorTarget[] => resolveConnectorTargetsWithCache(connectorId, index, templates, new Map());

const resolveConnectorTargetsWithCache = (
  connectorId: string,
  index: WorkflowGraphIndex,
  templates: InvocationTemplates | undefined,
  cache: Map<string, ResolvedConnectorTarget[]>
): ResolvedConnectorTarget[] => {
  const visited = new Set<string>();

  const resolve = (nodeId: string): ResolvedConnectorTarget[] => {
    const cached = cache.get(nodeId);

    if (cached) {
      return cached;
    }

    if (visited.has(nodeId)) {
      return [];
    }

    visited.add(nodeId);

    const targets: ResolvedConnectorTarget[] = [];

    for (const outboundEdge of getConnectorOutputEdgesIndexed(nodeId, index)) {
      const targetNode = index.nodesById.get(outboundEdge.target);

      if (!targetNode) {
        continue;
      }

      if (isInvocationNode(targetNode)) {
        targets.push({
          fieldName: outboundEdge.targetHandle,
          nodeId: targetNode.id,
          type: templates?.[targetNode.data.type]?.inputs[outboundEdge.targetHandle]?.type ?? null,
        });
        continue;
      }

      if (isConnectorNode(targetNode) && outboundEdge.targetHandle === CONNECTOR_INPUT_HANDLE) {
        targets.push(...resolve(targetNode.id));
      }
    }

    cache.set(nodeId, targets);
    return targets;
  };

  return resolve(connectorId);
};

export const resolveWorkflowEdgeSource = (
  edge: WorkflowEdge,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedConnectorSource | null =>
  resolveWorkflowEdgeSourceIndexed(edge, createWorkflowGraphIndex(nodes, edges), templates);

export const resolveWorkflowEdgeSourceIndexed = (
  edge: WorkflowEdge,
  index: WorkflowGraphIndex,
  templates?: InvocationTemplates,
  connectorSourceCache = new Map<string, ResolvedConnectorSource | null>()
): ResolvedConnectorSource | null => {
  const sourceNode = index.nodesById.get(edge.source);

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
    return resolveConnectorSourceWithCache(sourceNode.id, index, templates, connectorSourceCache);
  }

  return null;
};

export const getResolvedWorkflowEdges = (
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  templates?: InvocationTemplates
): ResolvedWorkflowEdge[] => getResolvedWorkflowEdgesIndexed(edges, createWorkflowGraphIndex(nodes, edges), templates);

export const getResolvedWorkflowEdgesIndexed = (
  edges: WorkflowEdge[],
  index: WorkflowGraphIndex,
  templates?: InvocationTemplates
): ResolvedWorkflowEdge[] => {
  const resolved: ResolvedWorkflowEdge[] = [];
  const connectorSourceCache = new Map<string, ResolvedConnectorSource | null>();

  for (const edge of edges) {
    const source = resolveWorkflowEdgeSourceIndexed(edge, index, templates, connectorSourceCache);

    if (!source) {
      continue;
    }

    resolved.push({ ...edge, source: source.nodeId, sourceHandle: source.fieldName });
  }

  return resolved;
};
