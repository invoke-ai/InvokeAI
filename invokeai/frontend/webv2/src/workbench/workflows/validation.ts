import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  InvocationTemplate,
  InvocationTemplates,
  ProjectGraphState,
  WorkflowEdge,
  WorkflowNode,
} from './types';

import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  resolveConnectorSource,
  resolveConnectorTargets,
} from './connectors';
import { isConnectorNode, isInvocationNode } from './types';

/**
 * Connection validation, ported from the legacy editor's
 * `validateConnectionTypes` / `validateConnection`. Connector nodes are
 * pass-through routing nodes: their source type is resolved from the first
 * upstream invocation output when one exists.
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

export const getCompatibleInputTemplate = (
  template: InvocationTemplate,
  sourceType: FieldType | null
): FieldInputTemplate | null => {
  const inputTemplates = Object.values(template.inputs).sort(
    (a, b) => (a.uiOrder ?? Number.MAX_SAFE_INTEGER) - (b.uiOrder ?? Number.MAX_SAFE_INTEGER)
  );

  return (
    inputTemplates.find(
      (inputTemplate) =>
        !inputTemplate.uiHidden &&
        inputTemplate.input !== 'direct' &&
        (sourceType === null || validateConnectionTypes(sourceType, inputTemplate.type))
    ) ?? null
  );
};

export const getCompatibleOutputTemplate = (
  template: InvocationTemplate,
  targetType: FieldType | null
): FieldOutputTemplate | null => {
  return (
    Object.values(template.outputs).find(
      (outputTemplate) => targetType === null || validateConnectionTypes(outputTemplate.type, targetType)
    ) ?? null
  );
};

const isSameFieldType = (left: FieldType, right: FieldType): boolean =>
  left.name === right.name && left.cardinality === right.cardinality && left.batch === right.batch;

const getCommonConnectorTargetType = (
  connectorId: string,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates
): FieldType | null => {
  const targets = resolveConnectorTargets(connectorId, document.nodes, document.edges, templates);

  if (targets.length === 0 || targets.some((target) => target.type === null)) {
    return null;
  }

  const firstType = targets[0]?.type;

  return firstType && targets.every((target) => target.type !== null && isSameFieldType(firstType, target.type))
    ? firstType
    : null;
};

const getSourceFieldType = (
  node: WorkflowNode,
  handle: string,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates
): FieldType | null | undefined => {
  if (isInvocationNode(node)) {
    return templates[node.data.type]?.outputs[handle]?.type;
  }

  if (isConnectorNode(node) && handle === CONNECTOR_OUTPUT_HANDLE) {
    const source = resolveConnectorSource(node.id, document.nodes, document.edges, templates);

    return source ? source.type : getCommonConnectorTargetType(node.id, document, templates);
  }

  return undefined;
};

export const getWorkflowSourceFieldType = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates,
  sourceNodeId: string,
  sourceHandle: string
): FieldType | null | undefined => {
  const sourceNode = document.nodes.find((node) => node.id === sourceNodeId);

  return sourceNode ? getSourceFieldType(sourceNode, sourceHandle, document, templates) : undefined;
};

const getTargetFieldType = (
  node: WorkflowNode,
  handle: string,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates
): FieldType | null | undefined => {
  if (isInvocationNode(node)) {
    const inputTemplate = templates[node.data.type]?.inputs[handle];

    return inputTemplate && inputTemplate.input !== 'direct' ? inputTemplate.type : undefined;
  }

  if (isConnectorNode(node) && handle === CONNECTOR_INPUT_HANDLE) {
    const targetType = getCommonConnectorTargetType(node.id, document, templates);

    if (targetType) {
      return targetType;
    }

    return resolveConnectorSource(node.id, document.nodes, document.edges, templates)?.type ?? null;
  }

  return undefined;
};

export const getWorkflowTargetFieldType = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates,
  targetNodeId: string,
  targetHandle: string
): FieldType | null | undefined => {
  const targetNode = document.nodes.find((node) => node.id === targetNodeId);

  return targetNode ? getTargetFieldType(targetNode, targetHandle, document, templates) : undefined;
};

const hasValidSourceHandle = (node: WorkflowNode, handle: string, templates: InvocationTemplates): boolean => {
  if (isInvocationNode(node)) {
    return templates[node.data.type]?.outputs[handle] !== undefined;
  }

  return isConnectorNode(node) && handle === CONNECTOR_OUTPUT_HANDLE;
};

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

  if (!sourceNode || !targetNode) {
    return 'Both ends of a connection must be workflow nodes.';
  }

  if (!hasValidSourceHandle(sourceNode, sourceHandle, templates)) {
    return 'One of the fields has no known definition.';
  }

  if (isConnectorNode(targetNode)) {
    if (targetHandle !== CONNECTOR_INPUT_HANDLE) {
      return 'Connectors only accept input on their left handle.';
    }

    if (document.edges.some((edge) => edge.target === targetNodeId && edge.targetHandle === targetHandle)) {
      return 'Connector already has an input.';
    }

    const sourceFieldType = getSourceFieldType(sourceNode, sourceHandle, document, templates);
    const targetFieldTypes = resolveConnectorTargets(targetNodeId, document.nodes, document.edges, templates)
      .map((target) => target.type)
      .filter((type): type is FieldType => type !== null);

    for (const targetFieldType of targetFieldTypes) {
      if (sourceFieldType && !validateConnectionTypes(sourceFieldType, targetFieldType)) {
        return `${sourceFieldType.name} cannot connect to ${targetFieldType.name}.`;
      }
    }

    if (wouldCreateCycle(sourceNodeId, targetNodeId, document.edges)) {
      return 'This connection would create a cycle.';
    }

    return null;
  }

  if (!isInvocationNode(targetNode)) {
    return 'The target node cannot receive connections.';
  }

  const targetTemplate = templates[targetNode.data.type];
  const sourceFieldType = getSourceFieldType(sourceNode, sourceHandle, document, templates);
  const targetField = targetTemplate?.inputs[targetHandle];

  if (sourceFieldType === undefined || !targetField) {
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

  if (sourceFieldType && !validateConnectionTypes(sourceFieldType, targetField.type)) {
    return `${sourceFieldType.name} cannot connect to ${targetField.type.name}.`;
  }

  if (wouldCreateCycle(sourceNodeId, targetNodeId, document.edges)) {
    return 'This connection would create a cycle.';
  }

  return null;
};
