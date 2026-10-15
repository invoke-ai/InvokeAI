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
  resolveConnectorSourceIndexed,
  resolveConnectorTargetsIndexed,
  resolveLoopLinkagePath,
} from './connectors';
import { createWorkflowGraphIndex, type WorkflowGraphIndex } from './graphIndex';
import { isConnectorNode, isInvocationNode } from './types';

export const LOOP_LINKAGE_FIELD = 'loop_linkage';

/** Resolve connector passthrough types from the first upstream invocation output before validating connections. */

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
      if (edge.type === 'loop_linkage') {
        continue;
      }
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
    if (edge.type === 'loop_linkage') {
      continue;
    }
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
        // Skip internal inputs for unresolved source guesses, but retain them for known matching types such as
        // MetadataField.
        !(sourceType === null && inputTemplate.fieldKind === 'internal') &&
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
  templates: InvocationTemplates,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): FieldType | null => {
  const targets = resolveConnectorTargetsIndexed(connectorId, index, templates);

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
  templates: InvocationTemplates,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): FieldType | null | undefined => {
  if (isInvocationNode(node)) {
    return templates[node.data.type]?.outputs[handle]?.type;
  }

  if (isConnectorNode(node) && handle === CONNECTOR_OUTPUT_HANDLE) {
    const source = resolveConnectorSourceIndexed(node.id, index, templates);

    return source ? source.type : getCommonConnectorTargetType(node.id, document, templates, index);
  }

  return undefined;
};

export const getWorkflowSourceFieldType = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates,
  sourceNodeId: string,
  sourceHandle: string,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): FieldType | null | undefined => {
  const sourceNode = index.nodesById.get(sourceNodeId);

  return sourceNode ? getSourceFieldType(sourceNode, sourceHandle, document, templates, index) : undefined;
};

const getTargetFieldType = (
  node: WorkflowNode,
  handle: string,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): FieldType | null | undefined => {
  if (isInvocationNode(node)) {
    const inputTemplate = node.data.dynamicInputTemplates?.[handle] ?? templates[node.data.type]?.inputs[handle];

    return inputTemplate && inputTemplate.input !== 'direct' ? inputTemplate.type : undefined;
  }

  if (isConnectorNode(node) && handle === CONNECTOR_INPUT_HANDLE) {
    const targetType = getCommonConnectorTargetType(node.id, document, templates, index);

    if (targetType) {
      return targetType;
    }

    return resolveConnectorSourceIndexed(node.id, index, templates)?.type ?? null;
  }

  return undefined;
};

export const getWorkflowTargetFieldType = (
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates,
  targetNodeId: string,
  targetHandle: string,
  index: WorkflowGraphIndex = createWorkflowGraphIndex(document.nodes, document.edges)
): FieldType | null | undefined => {
  const targetNode = index.nodesById.get(targetNodeId);

  return targetNode ? getTargetFieldType(targetNode, targetHandle, document, templates, index) : undefined;
};

const hasValidSourceHandle = (node: WorkflowNode, handle: string, templates: InvocationTemplates): boolean => {
  if (isInvocationNode(node)) {
    return templates[node.data.type]?.outputs[handle] !== undefined;
  }

  return isConnectorNode(node) && handle === CONNECTOR_OUTPUT_HANDLE;
};

const isLoopLinkageHandle = (handle: string): boolean => handle === LOOP_LINKAGE_FIELD;

interface LoopLinkageOwnership {
  forNodeId: string;
  returnNodeId: string;
}

const getLoopLinkageOwnerships = (document: Pick<ProjectGraphState, 'edges' | 'nodes'>): LoopLinkageOwnership[] => {
  const ownerships: LoopLinkageOwnership[] = [];

  for (const edge of document.edges) {
    if (
      edge.type === 'loop_linkage' &&
      edge.sourceHandle === LOOP_LINKAGE_FIELD &&
      edge.targetHandle === LOOP_LINKAGE_FIELD
    ) {
      ownerships.push({ forNodeId: edge.source, returnNodeId: edge.target });
      continue;
    }

    if (edge.type !== 'default' || edge.targetHandle !== LOOP_LINKAGE_FIELD) {
      continue;
    }

    const path = resolveLoopLinkagePath(edge, document.nodes, document.edges);
    if (path) {
      ownerships.push({ forNodeId: path.forNodeId, returnNodeId: path.returnNodeId });
    }
  }

  return ownerships;
};

const hasLoopLinkageOwnershipConflict = (
  candidate: LoopLinkageOwnership,
  existingOwnerships: LoopLinkageOwnership[]
): boolean =>
  existingOwnerships.some(
    (ownership) => ownership.forNodeId === candidate.forNodeId || ownership.returnNodeId === candidate.returnNodeId
  );

const getConnectorTerminalEdges = (connectorId: string, index: WorkflowGraphIndex): WorkflowEdge[] => {
  const pendingConnectorIds = [connectorId];
  const visitedConnectorIds = new Set<string>();
  const terminalEdges: WorkflowEdge[] = [];

  while (pendingConnectorIds.length > 0) {
    const currentConnectorId = pendingConnectorIds.pop();
    if (currentConnectorId === undefined || visitedConnectorIds.has(currentConnectorId)) {
      continue;
    }

    visitedConnectorIds.add(currentConnectorId);

    for (const edge of index.connectorOutputsById.get(currentConnectorId) ?? []) {
      const targetNode = index.nodesById.get(edge.target);
      if (targetNode && isConnectorNode(targetNode) && edge.targetHandle === CONNECTOR_INPUT_HANDLE) {
        pendingConnectorIds.push(targetNode.id);
      } else {
        terminalEdges.push(edge);
      }
    }
  }

  return terminalEdges;
};

const isValidLoopLinkageConnection = (
  sourceNode: WorkflowNode,
  sourceHandle: string,
  targetNode: WorkflowNode,
  targetHandle: string,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  index: WorkflowGraphIndex,
  templates: InvocationTemplates
): boolean => {
  const existingOwnerships = getLoopLinkageOwnerships(document);

  if (isInvocationNode(sourceNode) && isInvocationNode(targetNode)) {
    const isDirectLinkage =
      sourceNode.data.type === 'for' &&
      targetNode.data.type === 'for_return' &&
      sourceHandle === LOOP_LINKAGE_FIELD &&
      targetHandle === LOOP_LINKAGE_FIELD;

    return (
      isDirectLinkage &&
      !hasLoopLinkageOwnershipConflict({ forNodeId: sourceNode.id, returnNodeId: targetNode.id }, existingOwnerships)
    );
  }

  if (isInvocationNode(sourceNode) && isConnectorNode(targetNode)) {
    if (
      sourceNode.data.type === 'for' &&
      sourceHandle === LOOP_LINKAGE_FIELD &&
      targetHandle === CONNECTOR_INPUT_HANDLE
    ) {
      if (existingOwnerships.some((ownership) => ownership.forNodeId === sourceNode.id)) {
        return false;
      }

      const candidateEdge: WorkflowEdge = {
        id: '__candidate-loop-linkage__',
        source: sourceNode.id,
        sourceHandle,
        target: targetNode.id,
        targetHandle,
        type: 'default',
      };
      const terminalEdges = getConnectorTerminalEdges(targetNode.id, index);
      if (terminalEdges.length === 0) {
        return true;
      }

      const candidatePaths = terminalEdges.map((edge) =>
        resolveLoopLinkagePath(edge, document.nodes, [...document.edges, candidateEdge])
      );

      return (
        candidatePaths.length === 1 &&
        candidatePaths[0] !== null &&
        !hasLoopLinkageOwnershipConflict(candidatePaths[0], existingOwnerships)
      );
    }

    return false;
  }

  if (isConnectorNode(sourceNode) && isInvocationNode(targetNode)) {
    if (
      targetNode.data.type !== 'for_return' ||
      targetHandle !== LOOP_LINKAGE_FIELD ||
      sourceHandle !== CONNECTOR_OUTPUT_HANDLE
    ) {
      return false;
    }

    const resolvedSource = resolveConnectorSourceIndexed(sourceNode.id, index, templates);
    const resolvedSourceNode = resolvedSource ? index.nodesById.get(resolvedSource.nodeId) : undefined;

    if (existingOwnerships.some((ownership) => ownership.returnNodeId === targetNode.id)) {
      return false;
    }

    if (resolvedSource === null) {
      return true;
    }

    const candidateEdge: WorkflowEdge = {
      id: '__candidate-loop-linkage__',
      source: sourceNode.id,
      sourceHandle,
      target: targetNode.id,
      targetHandle,
      type: 'default',
    };
    const stagedEdges = [...document.edges, candidateEdge];
    const candidatePath = resolveLoopLinkagePath(candidateEdge, document.nodes, stagedEdges);

    return (
      resolvedSourceNode !== undefined &&
      isInvocationNode(resolvedSourceNode) &&
      resolvedSourceNode.data.type === 'for' &&
      resolvedSource.fieldName === LOOP_LINKAGE_FIELD &&
      candidatePath !== null &&
      !hasLoopLinkageOwnershipConflict(candidatePath, existingOwnerships)
    );
  }

  if (
    isConnectorNode(sourceNode) &&
    isConnectorNode(targetNode) &&
    sourceHandle === CONNECTOR_OUTPUT_HANDLE &&
    targetHandle === CONNECTOR_INPUT_HANDLE
  ) {
    const resolvedSource = resolveConnectorSourceIndexed(sourceNode.id, index, templates);
    if (resolvedSource?.fieldName !== LOOP_LINKAGE_FIELD) {
      return false;
    }

    const candidateEdge: WorkflowEdge = {
      id: '__candidate-loop-linkage__',
      source: sourceNode.id,
      sourceHandle,
      target: targetNode.id,
      targetHandle,
      type: 'default',
    };
    const terminalEdges = getConnectorTerminalEdges(targetNode.id, index);
    if (terminalEdges.length === 0) {
      return true;
    }

    const candidatePaths = terminalEdges.map((edge) =>
      resolveLoopLinkagePath(edge, document.nodes, [...document.edges, candidateEdge])
    );

    return (
      candidatePaths.length === 1 &&
      candidatePaths[0] !== null &&
      !hasLoopLinkageOwnershipConflict(candidatePaths[0], existingOwnerships)
    );
  }

  return false;
};

/** Returns a human-readable rejection reason, or null when the connection is valid. */
export const validateConnection = (
  candidate: ConnectionCandidate,
  document: Pick<ProjectGraphState, 'edges' | 'nodes'>,
  templates: InvocationTemplates
): string | null => {
  const { sourceHandle, sourceNodeId, targetHandle, targetNodeId } = candidate;
  const index = createWorkflowGraphIndex(document.nodes, document.edges);

  if (sourceNodeId === targetNodeId) {
    return 'A node cannot connect to itself.';
  }

  const sourceNode = index.nodesById.get(sourceNodeId);
  const targetNode = index.nodesById.get(targetNodeId);

  if (!sourceNode || !targetNode) {
    return 'Both ends of a connection must be workflow nodes.';
  }

  if (!hasValidSourceHandle(sourceNode, sourceHandle, templates)) {
    return 'One of the fields has no known definition.';
  }

  const sourceConnectorLoopLinkage =
    isConnectorNode(sourceNode) &&
    sourceHandle === CONNECTOR_OUTPUT_HANDLE &&
    resolveConnectorSourceIndexed(sourceNode.id, index, templates)?.fieldName === LOOP_LINKAGE_FIELD;

  if (isLoopLinkageHandle(sourceHandle) || isLoopLinkageHandle(targetHandle) || sourceConnectorLoopLinkage) {
    if (!isValidLoopLinkageConnection(sourceNode, sourceHandle, targetNode, targetHandle, document, index, templates)) {
      return 'For loop linkage must connect a For to its ForReturn.';
    }
  }

  if (isConnectorNode(targetNode)) {
    if (targetHandle !== CONNECTOR_INPUT_HANDLE) {
      return 'Connectors only accept input on their left handle.';
    }

    if (index.edgesByTarget.get(targetNodeId)?.some((edge) => edge.targetHandle === targetHandle)) {
      return 'Connector already has an input.';
    }

    const sourceFieldType = getSourceFieldType(sourceNode, sourceHandle, document, templates, index);
    const targetFieldTypes = resolveConnectorTargetsIndexed(targetNodeId, index, templates)
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
  const sourceFieldType = getSourceFieldType(sourceNode, sourceHandle, document, templates, index);
  const targetField = targetNode.data.dynamicInputTemplates?.[targetHandle] ?? targetTemplate?.inputs[targetHandle];

  if (sourceFieldType === undefined || !targetField) {
    return 'One of the fields has no known definition.';
  }

  if (targetField.input === 'direct') {
    return `${targetField.title} only accepts direct values, not connections.`;
  }

  const isDuplicate = index.edgesBySource
    .get(sourceNodeId)
    ?.some(
      (edge) =>
        edge.source === sourceNodeId &&
        edge.sourceHandle === sourceHandle &&
        edge.target === targetNodeId &&
        edge.targetHandle === targetHandle
    );

  if (isDuplicate) {
    return 'This connection already exists.';
  }

  const hasInboundEdge = index.edgesByTarget.get(targetNodeId)?.some((edge) => edge.targetHandle === targetHandle);

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
