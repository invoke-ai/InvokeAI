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

export interface ResolvedLoopLinkagePath {
  forNodeId: string;
  returnNodeId: string;
  edgeIds: string[];
  connectorNodeIds: string[];
}

export interface ConnectorDeletionSpliceConnection extends WorkflowEdge {
  type: 'default' | 'loop_linkage';
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

/** Resolves the optional connector representation of a For-to-ForReturn linkage. */
export const resolveLoopLinkagePath = (
  edge: WorkflowEdge,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): ResolvedLoopLinkagePath | null => {
  if (edge.type !== 'default' || edge.targetHandle !== 'loop_linkage') {
    return null;
  }

  const returnNode = nodes.find((node) => node.id === edge.target);

  if (!returnNode || !isInvocationNode(returnNode) || returnNode.data.type !== 'for_return') {
    return null;
  }

  const edgeIds = [edge.id];
  const connectorNodeIds: string[] = [];
  const visitedConnectors = new Set<string>();
  let currentEdge = edge;

  while (true) {
    const sourceNode = nodes.find((node) => node.id === currentEdge.source);

    if (!sourceNode) {
      return null;
    }

    if (sourceNode && isInvocationNode(sourceNode)) {
      if (sourceNode.data.type !== 'for' || currentEdge.sourceHandle !== 'loop_linkage') {
        return null;
      }

      return {
        connectorNodeIds: [...connectorNodeIds].reverse(),
        edgeIds: [...edgeIds].reverse(),
        forNodeId: sourceNode.id,
        returnNodeId: returnNode.id,
      };
    }

    if (!isConnectorNode(sourceNode) || currentEdge.sourceHandle !== CONNECTOR_OUTPUT_HANDLE) {
      return null;
    }

    if (visitedConnectors.has(sourceNode.id)) {
      return null;
    }

    visitedConnectors.add(sourceNode.id);
    connectorNodeIds.push(sourceNode.id);

    const inputEdges = edges.filter(
      (candidate) =>
        candidate.type === 'default' &&
        candidate.target === sourceNode.id &&
        candidate.targetHandle === CONNECTOR_INPUT_HANDLE
    );
    const outputEdges = getConnectorOutputEdges(sourceNode.id, edges);

    if (inputEdges.length !== 1 || outputEdges.length !== 1 || outputEdges[0]?.id !== currentEdge.id) {
      return null;
    }

    const inputEdge = inputEdges[0];

    if (!inputEdge) {
      return null;
    }

    edgeIds.push(inputEdge.id);
    currentEdge = inputEdge;
  }
};

/** Returns document edges with complete connector aliases styled as loop linkages. */
export const getEdgesWithLoopLinkageAliases = (nodes: WorkflowNode[], edges: WorkflowEdge[]): WorkflowEdge[] => {
  const linkageEdgeIds = new Set<string>();

  for (const edge of edges) {
    const sourceNode = nodes.find((node) => node.id === edge.source);

    if (
      edge.type !== 'default' ||
      edge.sourceHandle !== CONNECTOR_OUTPUT_HANDLE ||
      edge.targetHandle !== 'loop_linkage' ||
      !sourceNode ||
      !isConnectorNode(sourceNode)
    ) {
      continue;
    }

    resolveLoopLinkagePath(edge, nodes, edges)?.edgeIds.forEach((edgeId) => linkageEdgeIds.add(edgeId));
  }

  return edges.map((edge) => (linkageEdgeIds.has(edge.id) ? { ...edge, type: 'loop_linkage' } : edge));
};

const resolveConnectorDeletionSource = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  removedConnectorIds: ReadonlySet<string>
): ResolvedConnectorSource | null => {
  const visitedConnectorIds = new Set<string>();
  let resolvedSource = resolveConnectorSource(connectorId, nodes, edges);

  while (resolvedSource && removedConnectorIds.has(resolvedSource.nodeId)) {
    if (visitedConnectorIds.has(resolvedSource.nodeId)) {
      return null;
    }
    visitedConnectorIds.add(resolvedSource.nodeId);
    resolvedSource = resolveConnectorSource(resolvedSource.nodeId, nodes, edges);
  }

  if (!resolvedSource) {
    return null;
  }

  const sourceNode = nodes.find((node) => node.id === resolvedSource.nodeId);
  if (
    sourceNode &&
    isInvocationNode(sourceNode) &&
    sourceNode.data.type === 'for' &&
    resolvedSource.fieldName === 'loop_linkage'
  ) {
    const outputEdges = getConnectorOutputEdges(connectorId, edges);
    if (
      outputEdges.length !== 1 ||
      outputEdges.some((edge) => {
        const targetNode = nodes.find((node) => node.id === edge.target);
        return !(
          (targetNode && isConnectorNode(targetNode) && edge.targetHandle === CONNECTOR_INPUT_HANDLE) ||
          (targetNode &&
            isInvocationNode(targetNode) &&
            targetNode.data.type === 'for_return' &&
            edge.targetHandle === 'loop_linkage')
        );
      })
    ) {
      return null;
    }
  }

  const linkagePath = getResolvedLoopLinkagePathForConnector(connectorId, nodes, edges);
  const inputEdge = getConnectorInputEdge(connectorId, edges);
  if (linkagePath && inputEdge) {
    return { fieldName: inputEdge.sourceHandle, nodeId: inputEdge.source, type: null };
  }

  return resolvedSource;
};

const getResolvedLoopLinkagePathForConnector = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): ResolvedLoopLinkagePath | null => {
  for (const edge of edges) {
    if (
      edge.type !== 'default' ||
      edge.sourceHandle !== CONNECTOR_OUTPUT_HANDLE ||
      edge.targetHandle !== 'loop_linkage'
    ) {
      continue;
    }

    const path = resolveLoopLinkagePath(edge, nodes, edges);
    if (path?.connectorNodeIds.includes(connectorId)) {
      return path;
    }
  }

  return null;
};

const getConnectorDeletionOutputEdges = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  removedConnectorIds: ReadonlySet<string>
): WorkflowEdge[] | null => {
  const visited = new Set<string>();
  const outputEdges: WorkflowEdge[] = [];

  const visit = (currentConnectorId: string): boolean => {
    if (visited.has(currentConnectorId)) {
      return false;
    }

    visited.add(currentConnectorId);

    for (const edge of getConnectorOutputEdges(currentConnectorId, edges)) {
      const targetNode = nodes.find((node) => node.id === edge.target);

      if (targetNode && isConnectorNode(targetNode) && removedConnectorIds.has(targetNode.id)) {
        if (!visit(targetNode.id)) {
          return false;
        }
      } else {
        outputEdges.push(edge);
      }
    }

    return true;
  };

  return visit(connectorId) ? outputEdges : null;
};

/** Builds replacement edges before deleting one or more connector nodes. */
export const getConnectorDeletionSpliceConnections = (
  connectorId: string,
  nodes: WorkflowNode[],
  edges: WorkflowEdge[],
  removedConnectorIds: ReadonlySet<string> = new Set([connectorId])
): ConnectorDeletionSpliceConnection[] => {
  const source = resolveConnectorDeletionSource(connectorId, nodes, edges, removedConnectorIds);
  const outputEdges = getConnectorDeletionOutputEdges(connectorId, nodes, edges, removedConnectorIds);

  if (!source || !outputEdges) {
    return [];
  }

  const connections: ConnectorDeletionSpliceConnection[] = [];

  for (const outputEdge of outputEdges) {
    const type =
      source.fieldName === 'loop_linkage' && outputEdge.targetHandle === 'loop_linkage' ? 'loop_linkage' : 'default';
    const replacement = {
      id: `splice-${source.nodeId}-${source.fieldName}-${outputEdge.target}-${outputEdge.targetHandle}`,
      source: source.nodeId,
      sourceHandle: source.fieldName,
      target: outputEdge.target,
      targetHandle: outputEdge.targetHandle,
      type,
    } satisfies ConnectorDeletionSpliceConnection;

    if (
      !connections.some(
        (edge) =>
          edge.source === replacement.source &&
          edge.sourceHandle === replacement.sourceHandle &&
          edge.target === replacement.target &&
          edge.targetHandle === replacement.targetHandle
      )
    ) {
      connections.push(replacement);
    }
  }

  return connections;
};

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
          type:
            targetNode.data.dynamicInputTemplates?.[outboundEdge.targetHandle]?.type ??
            templates?.[targetNode.data.type]?.inputs[outboundEdge.targetHandle]?.type ??
            null,
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
