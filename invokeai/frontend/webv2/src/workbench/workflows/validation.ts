import type { FieldType, InvocationTemplates, ProjectGraphState, WorkflowEdge, WorkflowNode } from './types';
import { isInvocationNode } from './types';

/**
 * Connection validation, ported from the legacy editor's
 * `validateConnectionTypes` / `validateConnection`. Connector and batch node
 * special cases are intentionally absent — those node kinds are not part of
 * the v7 document model.
 */

const isSingle = (type: FieldType): boolean => type.cardinality === 'SINGLE';
const isCollection = (type: FieldType): boolean => type.cardinality === 'COLLECTION';
const isSingleOrCollection = (type: FieldType): boolean => type.cardinality === 'SINGLE_OR_COLLECTION';

const isSameShape = (a: FieldType, b: FieldType): boolean =>
  a.name === b.name && a.cardinality === b.cardinality && a.batch === b.batch;

/** Types are equal when either declared or `ui_type`-original sides match. */
export const areFieldTypesEqual = (a: FieldType, b: FieldType): boolean =>
  isSameShape(a, b) ||
  (b.originalType !== undefined && isSameShape(a, b.originalType)) ||
  (a.originalType !== undefined && isSameShape(a.originalType, b)) ||
  (a.originalType !== undefined && b.originalType !== undefined && isSameShape(a.originalType, b.originalType));

export const validateConnectionTypes = (sourceType: FieldType, targetType: FieldType): boolean => {
  if (areFieldTypesEqual(sourceType, targetType)) {
    return true;
  }

  if (sourceType.batch !== targetType.batch) {
    return false;
  }

  const isCollectionItemToNonCollection = sourceType.name === 'CollectionItemField' && !isCollection(targetType);
  const isNonCollectionToCollectionItem = isSingle(sourceType) && targetType.name === 'CollectionItemField';
  const isAnythingToSingleOrCollectionOfSameBaseType =
    isSingleOrCollection(targetType) && sourceType.name === targetType.name;
  const isGenericCollectionToAnyCollectionOrSingleOrCollection =
    sourceType.name === 'CollectionField' && !isSingle(targetType);
  const isCollectionToGenericCollection = targetType.name === 'CollectionField' && isCollection(sourceType);

  const doesCardinalityMatch =
    (isSingle(sourceType) && isSingle(targetType)) ||
    (isCollection(sourceType) && isCollection(targetType)) ||
    (isCollection(sourceType) && isSingleOrCollection(targetType)) ||
    (isSingleOrCollection(sourceType) && isSingleOrCollection(targetType)) ||
    (isSingle(sourceType) && isSingleOrCollection(targetType));

  const isIntToFloat = sourceType.name === 'IntegerField' && targetType.name === 'FloatField';
  const isIntToString = sourceType.name === 'IntegerField' && targetType.name === 'StringField';
  const isFloatToString = sourceType.name === 'FloatField' && targetType.name === 'StringField';
  const isSubTypeMatch = doesCardinalityMatch && (isIntToFloat || isIntToString || isFloatToString);

  const isTargetAnyType = targetType.name === 'AnyField';
  const isSourceAnyType = sourceType.name === 'AnyField' && doesCardinalityMatch;

  return (
    isCollectionItemToNonCollection ||
    isNonCollectionToCollectionItem ||
    isAnythingToSingleOrCollectionOfSameBaseType ||
    isGenericCollectionToAnyCollectionOrSingleOrCollection ||
    isCollectionToGenericCollection ||
    isSubTypeMatch ||
    isTargetAnyType ||
    isSourceAnyType
  );
};

/** True if adding source→target would close a cycle: target must not already reach source. */
export const wouldCreateCycle = (sourceNodeId: string, targetNodeId: string, edges: WorkflowEdge[]): boolean => {
  if (sourceNodeId === targetNodeId) {
    return true;
  }

  const visited = new Set<string>();
  const stack = [targetNodeId];

  while (stack.length > 0) {
    const nodeId = stack.pop() as string;

    if (nodeId === sourceNodeId) {
      return true;
    }

    if (visited.has(nodeId)) {
      continue;
    }

    visited.add(nodeId);

    for (const edge of edges) {
      if (edge.source === nodeId) {
        stack.push(edge.target);
      }
    }
  }

  return false;
};

/** True if the graph already contains a cycle anywhere. */
export const hasAnyCycle = (nodes: WorkflowNode[], edges: WorkflowEdge[]): boolean => {
  const adjacency = new Map<string, string[]>();

  for (const edge of edges) {
    adjacency.set(edge.source, [...(adjacency.get(edge.source) ?? []), edge.target]);
  }

  const state = new Map<string, 'visiting' | 'done'>();

  const visit = (nodeId: string): boolean => {
    const nodeState = state.get(nodeId);

    if (nodeState === 'visiting') {
      return true;
    }

    if (nodeState === 'done') {
      return false;
    }

    state.set(nodeId, 'visiting');

    for (const nextNodeId of adjacency.get(nodeId) ?? []) {
      if (visit(nextNodeId)) {
        return true;
      }
    }

    state.set(nodeId, 'done');

    return false;
  };

  return nodes.some((node) => visit(node.id));
};

export interface ConnectionCandidate {
  sourceNodeId: string;
  sourceHandle: string;
  targetNodeId: string;
  targetHandle: string;
}

/** Multiple inbound edges are only meaningful for the collect node's `item` input. */
const allowsMultipleInboundEdges = (node: WorkflowNode, fieldName: string): boolean =>
  isInvocationNode(node) && node.data.type === 'collect' && fieldName === 'item';

/** Returns a human-readable rejection reason, or null when the connection is valid. */
export const validateConnection = (
  candidate: ConnectionCandidate,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates
): string | null => {
  const { sourceHandle, sourceNodeId, targetHandle, targetNodeId } = candidate;

  if (sourceNodeId === targetNodeId) {
    return 'A node cannot connect to itself.';
  }

  const sourceNode = document.nodes.find((node) => node.id === sourceNodeId);
  const targetNode = document.nodes.find((node) => node.id === targetNodeId);

  if (!sourceNode || !isInvocationNode(sourceNode) || !targetNode || !isInvocationNode(targetNode)) {
    return 'Both ends of a connection must be invocation nodes.';
  }

  const sourceTemplate = templates[sourceNode.data.type];
  const targetTemplate = templates[targetNode.data.type];
  const sourceField = sourceTemplate?.outputs[sourceHandle];
  const targetField = targetTemplate?.inputs[targetHandle];

  if (!sourceField || !targetField) {
    return 'One of the fields has no known definition.';
  }

  if (targetField.input === 'direct') {
    return `${targetField.title} only accepts direct values, not connections.`;
  }

  const isDuplicate = document.edges.some(
    (edge) =>
      edge.source === sourceNodeId &&
      edge.sourceHandle === sourceHandle &&
      edge.target === targetNodeId &&
      edge.targetHandle === targetHandle
  );

  if (isDuplicate) {
    return 'This connection already exists.';
  }

  const hasInboundEdge = document.edges.some(
    (edge) => edge.target === targetNodeId && edge.targetHandle === targetHandle
  );

  if (hasInboundEdge && !allowsMultipleInboundEdges(targetNode, targetHandle)) {
    return `${targetField.title} already has a connection.`;
  }

  if (!validateConnectionTypes(sourceField.type, targetField.type)) {
    return `${sourceField.type.name} cannot connect to ${targetField.type.name}.`;
  }

  if (wouldCreateCycle(sourceNodeId, targetNodeId, document.edges)) {
    return 'This connection would create a cycle.';
  }

  return null;
};
