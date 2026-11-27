import type { Templates } from 'features/nodes/store/types';
import type { FieldType } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
import { isConnectorNode, isInvocationNode } from 'features/nodes/types/invocation';

export const CONNECTOR_INPUT_HANDLE = 'in';
export const CONNECTOR_OUTPUT_HANDLE = 'out';

type ResolvedConnectorSource = {
  nodeId: string;
  fieldName: string;
};

type SpliceConnection = {
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
};

type SpliceConnectionValidator = (
  connection: SpliceConnection,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates,
  ignoreEdge: AnyEdge | null,
  strict?: boolean
) => string | null;

export const getConnectorInputEdge = (connectorId: string, edges: AnyEdge[]): AnyEdge | undefined =>
  edges.find(
    (edge) =>
      edge.type === 'default' &&
      edge.target === connectorId &&
      edge.targetHandle === CONNECTOR_INPUT_HANDLE &&
      typeof edge.sourceHandle === 'string'
  );

export const getConnectorOutputEdges = (connectorId: string, edges: AnyEdge[]): AnyEdge[] =>
  edges.filter(
    (edge) =>
      edge.type === 'default' &&
      edge.source === connectorId &&
      edge.sourceHandle === CONNECTOR_OUTPUT_HANDLE &&
      typeof edge.targetHandle === 'string'
  );

export const resolveConnectorSource = (
  connectorId: string,
  nodes: AnyNode[],
  edges: AnyEdge[]
): ResolvedConnectorSource | null => {
  const visited = new Set<string>();

  const resolve = (nodeId: string): ResolvedConnectorSource | null => {
    if (visited.has(nodeId)) {
      return null;
    }
    visited.add(nodeId);

    const incomingEdge = getConnectorInputEdge(nodeId, edges);
    if (!incomingEdge || incomingEdge.type !== 'default') {
      return null;
    }
    if (typeof incomingEdge.sourceHandle !== 'string') {
      return null;
    }

    const sourceNode = nodes.find((node) => node.id === incomingEdge.source);
    if (!sourceNode) {
      return null;
    }

    if (isInvocationNode(sourceNode)) {
      return { nodeId: sourceNode.id, fieldName: incomingEdge.sourceHandle };
    }

    if (isConnectorNode(sourceNode)) {
      return resolve(sourceNode.id);
    }

    return null;
  };

  return resolve(connectorId);
};

export const resolveConnectorSourceFieldType = (
  connectorId: string,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates
): FieldType | null => {
  const resolvedSource = resolveConnectorSource(connectorId, nodes, edges);
  if (!resolvedSource) {
    return null;
  }

  const sourceNode = nodes.find((node) => node.id === resolvedSource.nodeId);
  if (!sourceNode || !isInvocationNode(sourceNode)) {
    return null;
  }

  const sourceTemplate = templates[sourceNode.data.type];
  return sourceTemplate?.outputs[resolvedSource.fieldName]?.type ?? null;
};

export const getConnectorDeletionSpliceConnections = (
  connectorId: string,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates,
  validateConnection?: SpliceConnectionValidator
): SpliceConnection[] | null => {
  const resolvedSource = resolveConnectorSource(connectorId, nodes, edges);
  if (!resolvedSource) {
    return null;
  }

  const outputEdges = getConnectorOutputEdges(connectorId, edges);
  const spliceConnections = outputEdges
    .filter((edge): edge is AnyEdge & { type: 'default'; targetHandle: string } => edge.type === 'default')
    .map((edge) => ({
      source: resolvedSource.nodeId,
      sourceHandle: resolvedSource.fieldName,
      target: edge.target,
      targetHandle: edge.targetHandle,
    }));

  const deduped = new Set<string>();
  for (const connection of spliceConnections) {
    const key = `${connection.source}:${connection.sourceHandle}->${connection.target}:${connection.targetHandle}`;
    if (deduped.has(key)) {
      return null;
    }
    deduped.add(key);
  }

  if (!validateConnection) {
    const sourceType = resolveConnectorSourceFieldType(connectorId, nodes, edges, templates);
    if (!sourceType) {
      return null;
    }
    const inputEdgeId = getConnectorInputEdge(connectorId, edges)?.id;
    const outputEdgeIds = new Set(outputEdges.map((edge) => edge.id));

    for (const connection of spliceConnections) {
      const targetNode = nodes.find((node) => node.id === connection.target);
      if (!targetNode || !isInvocationNode(targetNode)) {
        return null;
      }
      const targetTemplate = templates[targetNode.data.type];
      const targetFieldTemplate = targetTemplate?.inputs[connection.targetHandle];
      if (!targetFieldTemplate) {
        return null;
      }

      const matchesExistingDirectEdge = edges.some(
        (edge) =>
          edge.type === 'default' &&
          edge.source === connection.source &&
          edge.sourceHandle === connection.sourceHandle &&
          edge.target === connection.target &&
          edge.targetHandle === connection.targetHandle
      );
      if (matchesExistingDirectEdge) {
        return null;
      }

      const targetConflictCount = spliceConnections.filter(
        (candidate) => candidate.target === connection.target && candidate.targetHandle === connection.targetHandle
      ).length;
      const existingTargetConflict = edges.some(
        (edge) =>
          edge.type === 'default' &&
          edge.id !== inputEdgeId &&
          !outputEdgeIds.has(edge.id) &&
          edge.target === connection.target &&
          edge.targetHandle === connection.targetHandle
      );
      if (
        targetFieldTemplate.type.name !== 'CollectionItemField' &&
        (targetConflictCount > 1 || existingTargetConflict)
      ) {
        return null;
      }

      if (
        sourceType.name !== targetFieldTemplate.type.name &&
        targetFieldTemplate.type.name !== 'CollectionItemField'
      ) {
        return null;
      }
    }

    return spliceConnections;
  }

  const ignoredEdgeIds = new Set([
    getConnectorInputEdge(connectorId, edges)?.id,
    ...outputEdges.map((edge) => edge.id),
  ]);
  const existingEdges = edges.filter((edge) => !ignoredEdgeIds.has(edge.id));
  const stagedConnections: SpliceConnection[] = [];

  for (const connection of spliceConnections) {
    const stagedEdges = [
      ...existingEdges,
      ...stagedConnections.map(
        ({ source, sourceHandle, target, targetHandle }) =>
          ({
            id: `splice-${source}-${sourceHandle}-${target}-${targetHandle}`,
            type: 'default',
            source,
            sourceHandle,
            target,
            targetHandle,
          }) satisfies AnyEdge
      ),
    ];
    if (validateConnection(connection, nodes, stagedEdges, templates, null, true) !== null) {
      return null;
    }
    stagedConnections.push(connection);
  }

  return spliceConnections;
};
