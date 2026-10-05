import type { WorkflowEdge, WorkflowNode, ProjectGraphState } from './types';

import { getResolvedWorkflowEdgesIndexed, resolveLoopLinkagePath } from './connectors';
import { createWorkflowGraphIndex } from './graphIndex';
import { isInvocationNode } from './types';
import { LOOP_LINKAGE_FIELD } from './validation';

const FOR_LOOP_GRAPH_ERROR_CODES = [
  'nodes.forLoopMissingIterationOutput',
  'nodes.forLoopReturnCount',
  'nodes.forLoopUnterminatedBody',
  'nodes.forLoopNestedUnsupported',
  'nodes.forLoopIterateUnsupported',
  'nodes.forLoopIteratorInputUnsupported',
  'nodes.forLoopFinalOutputInBody',
  'nodes.forLoopBodyEscape',
  'nodes.forLoopInputCount',
  'nodes.forReturnInputCount',
  'nodes.forLoopLinkageMissing',
  'nodes.forLoopLinkageInvalid',
  'nodes.forLoopLinkageDuplicate',
  'nodes.forReturnOwnership',
] as const;

export type ForLoopGraphError = (typeof FOR_LOOP_GRAPH_ERROR_CODES)[number];

export interface ForLoopValidationReason {
  key: ForLoopGraphError;
}

export const createForLoopValidationReason = (key: ForLoopGraphError): ForLoopValidationReason => ({ key });

export class ForLoopGraphValidationError extends Error {
  readonly reason: ForLoopValidationReason;

  constructor(key: ForLoopGraphError) {
    super(`For loop validation failed: ${key}.`);
    this.name = 'ForLoopGraphValidationError';
    this.reason = createForLoopValidationReason(key);
  }
}

export const localizeForLoopValidationReason = (
  reason: string | ForLoopValidationReason,
  translate: (key: string) => string
): string => {
  if (typeof reason === 'string') {
    return reason;
  }

  return `${translate('nodes.forLoopValidationFailed')}: ${translate(reason.key)}.`;
};

export type LoopBodyBoundaryStatus =
  | 'complete'
  | 'missing_linkage'
  | 'invalid_linkage'
  | 'duplicate_linkage'
  | 'missing_return'
  | 'multiple_returns'
  | 'orphan_return';

export interface LoopBodyBoundary {
  forNodeId?: string;
  returnNodeId?: string;
  bodyNodeIds: string[];
  status: LoopBodyBoundaryStatus;
}

const ITERATION_OUTPUT_FIELDS = new Set(['item', 'index', 'total', 'state']);
const FINAL_OUTPUT_FIELDS = new Set(['output_collection', 'final_state']);

export const shouldAddForReturnLoopLinkage = (
  targetType: string,
  resolvedSource: { fieldName: string; nodeId: string } | null,
  resolvedSourceNode: WorkflowNode | undefined,
  edges: readonly WorkflowEdge[]
): boolean =>
  targetType === 'for_return' &&
  ITERATION_OUTPUT_FIELDS.has(resolvedSource?.fieldName ?? '') &&
  resolvedSourceNode?.type === 'invocation' &&
  resolvedSourceNode.data.type === 'for' &&
  !edges.some((edge) => edge.source === resolvedSourceNode.id && edge.sourceHandle === LOOP_LINKAGE_FIELD);

type LoopNode = { id: string; type: string };
type LoopEdge = {
  id: string;
  type: WorkflowEdge['type'];
  source: { node_id: string; field: string };
  destination: { node_id: string; field: string };
};

const getInvocationNodes = (nodes: WorkflowNode[]): LoopNode[] =>
  nodes.filter(isInvocationNode).map((node) => ({ id: node.id, type: node.data.type }));

/** Resolves connectors for loop validation and compilation without mutating the document. */
export const getCanonicalWorkflowEdges = (document: Pick<ProjectGraphState, 'nodes' | 'edges'>): LoopEdge[] => {
  const index = createWorkflowGraphIndex(document.nodes, document.edges);
  const aliasEdgeIds = new Set<string>();
  const aliasEdges: LoopEdge[] = [];

  for (const edge of document.edges) {
    const sourceNode = index.nodesById.get(edge.source);

    if (
      edge.type !== 'default' ||
      edge.targetHandle !== LOOP_LINKAGE_FIELD ||
      !sourceNode ||
      isInvocationNode(sourceNode)
    ) {
      continue;
    }

    const path = resolveLoopLinkagePath(edge, document.nodes, document.edges);

    if (!path) {
      continue;
    }

    path.edgeIds.forEach((edgeId) => aliasEdgeIds.add(edgeId));
    aliasEdges.push({
      destination: { field: LOOP_LINKAGE_FIELD, node_id: path.returnNodeId },
      id: `resolved-loop-linkage-${path.forNodeId}-${path.returnNodeId}`,
      source: { field: LOOP_LINKAGE_FIELD, node_id: path.forNodeId },
      type: 'loop_linkage',
    });
  }

  const resolved = getResolvedWorkflowEdgesIndexed(document.edges, index).filter(
    (edge) =>
      !aliasEdgeIds.has(edge.id) &&
      index.nodesById.get(edge.source) !== undefined &&
      index.nodesById.get(edge.target) !== undefined &&
      isInvocationNode(index.nodesById.get(edge.source) as WorkflowNode) &&
      isInvocationNode(index.nodesById.get(edge.target) as WorkflowNode)
  );

  return [
    ...resolved.map((edge) => ({
      destination: { field: edge.targetHandle, node_id: edge.target },
      id: edge.id,
      source: { field: edge.sourceHandle, node_id: edge.source },
      type: edge.type,
    })),
    ...aliasEdges,
  ];
};

const walk = (startIds: Iterable<string>, adjacency: Map<string, string[]>): Set<string> => {
  const visited = new Set<string>();
  const pending = [...startIds];

  while (pending.length > 0) {
    const nodeId = pending.pop();

    if (nodeId === undefined || visited.has(nodeId)) {
      continue;
    }

    visited.add(nodeId);
    pending.push(...(adjacency.get(nodeId) ?? []));
  }

  return visited;
};

const getBodyNodeIds = (
  reachableNodeIds: Set<string>,
  returnNodeId: string | undefined,
  incoming: Map<string, string[]>
): Set<string> => {
  if (!returnNodeId) {
    return new Set(reachableNodeIds);
  }

  const bodyNodeIds = new Set([...walk([returnNodeId], incoming)].filter((nodeId) => reachableNodeIds.has(nodeId)));
  bodyNodeIds.add(returnNodeId);
  return bodyNodeIds;
};

export const getForLoopBodyBoundaries = (nodes: WorkflowNode[], edges: WorkflowEdge[]): LoopBodyBoundary[] => {
  const invocationNodes = nodes.filter(isInvocationNode);
  const nodesById = new Map(invocationNodes.map((node) => [node.id, node]));
  const canonicalEdges = getCanonicalWorkflowEdges({ nodes, edges });
  const dataEdges = canonicalEdges.filter((edge) => edge.type !== 'loop_linkage');
  const outgoing = new Map<string, string[]>();
  const incoming = new Map<string, string[]>();

  for (const edge of dataEdges) {
    outgoing.set(edge.source.node_id, [...(outgoing.get(edge.source.node_id) ?? []), edge.destination.node_id]);
    incoming.set(edge.destination.node_id, [...(incoming.get(edge.destination.node_id) ?? []), edge.source.node_id]);
  }

  const linkedReturnByForId = new Map<string, string>();
  const linkedForByReturnId = new Map<string, string>();
  const duplicateForIds = new Set<string>();
  const duplicateReturnIds = new Set<string>();

  for (const edge of canonicalEdges.filter((candidate) => candidate.type === 'loop_linkage')) {
    const existingForId = linkedForByReturnId.get(edge.destination.node_id);
    if (linkedReturnByForId.has(edge.source.node_id) || existingForId !== undefined) {
      duplicateForIds.add(edge.source.node_id);
      if (existingForId !== undefined) {
        duplicateForIds.add(existingForId);
        duplicateReturnIds.add(edge.destination.node_id);
      }
      continue;
    }
    linkedReturnByForId.set(edge.source.node_id, edge.destination.node_id);
    linkedForByReturnId.set(edge.destination.node_id, edge.source.node_id);
  }

  const reachableReturnIds = new Set<string>();
  const boundaries = invocationNodes
    .filter((node) => node.data.type === 'for')
    .map<LoopBodyBoundary>((forNode) => {
      const iterationTargets = dataEdges
        .filter((edge) => edge.source.node_id === forNode.id && ITERATION_OUTPUT_FIELDS.has(edge.source.field))
        .map((edge) => edge.destination.node_id);
      const reachableNodeIds = walk(iterationTargets, outgoing);
      const reachableReturns = [...reachableNodeIds].filter(
        (nodeId) => nodesById.get(nodeId)?.data.type === 'for_return'
      );
      reachableReturns.forEach((nodeId) => reachableReturnIds.add(nodeId));
      const linkedReturnId = linkedReturnByForId.get(forNode.id);
      const returnNodeId = linkedReturnId ?? (reachableReturns.length === 1 ? reachableReturns[0] : undefined);
      let status: LoopBodyBoundaryStatus;

      if (duplicateForIds.has(forNode.id) || (linkedReturnId !== undefined && duplicateReturnIds.has(linkedReturnId))) {
        status = 'duplicate_linkage';
      } else if (linkedReturnId === undefined) {
        status = 'missing_linkage';
      } else if (!reachableNodeIds.has(linkedReturnId)) {
        status = 'invalid_linkage';
      } else if (reachableReturns.length === 0) {
        status = 'missing_return';
      } else if (reachableReturns.length > 1) {
        status = 'multiple_returns';
      } else {
        status = 'complete';
      }

      return {
        ...(returnNodeId ? { returnNodeId } : {}),
        bodyNodeIds: [forNode.id, ...getBodyNodeIds(reachableNodeIds, returnNodeId, incoming)],
        forNodeId: forNode.id,
        status,
      };
    });

  boundaries.push(
    ...invocationNodes
      .filter(
        (node) =>
          node.data.type === 'for_return' && !linkedForByReturnId.has(node.id) && !reachableReturnIds.has(node.id)
      )
      .map((node) => ({
        bodyNodeIds: [node.id],
        returnNodeId: node.id,
        status: 'orphan_return' as const,
      }))
  );

  return boundaries;
};

/** Validates the scheduler-specific For/ForReturn graph contract before queueing. */
export const validateForLoopGraph = (
  document: Pick<ProjectGraphState, 'nodes' | 'edges'>
): ForLoopGraphError | null => {
  const nodes = getInvocationNodes(document.nodes);
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const allEdges = getCanonicalWorkflowEdges(document);

  if (
    allEdges.some(
      (edge) =>
        edge.type === 'default' &&
        (edge.source.field === LOOP_LINKAGE_FIELD || edge.destination.field === LOOP_LINKAGE_FIELD) &&
        nodesById.has(edge.source.node_id) &&
        nodesById.has(edge.destination.node_id)
    )
  ) {
    return 'nodes.forLoopLinkageInvalid';
  }

  const edges = allEdges.filter((edge) => edge.type !== 'loop_linkage');
  const linkageEdges = allEdges.filter((edge) => edge.type === 'loop_linkage');
  const outgoing = new Map<string, string[]>();
  const incoming = new Map<string, string[]>();

  for (const edge of edges) {
    outgoing.set(edge.source.node_id, [...(outgoing.get(edge.source.node_id) ?? []), edge.destination.node_id]);
    incoming.set(edge.destination.node_id, [...(incoming.get(edge.destination.node_id) ?? []), edge.source.node_id]);
  }

  const linkedReturnByForId = new Map<string, string>();
  const linkedForByReturnId = new Map<string, string>();

  for (const edge of linkageEdges) {
    const sourceNode = nodesById.get(edge.source.node_id);
    const returnNode = nodesById.get(edge.destination.node_id);

    if (
      edge.source.field !== LOOP_LINKAGE_FIELD ||
      edge.destination.field !== LOOP_LINKAGE_FIELD ||
      sourceNode?.type !== 'for' ||
      returnNode?.type !== 'for_return'
    ) {
      return 'nodes.forLoopLinkageInvalid';
    }

    if (linkedReturnByForId.has(edge.source.node_id) || linkedForByReturnId.has(edge.destination.node_id)) {
      return 'nodes.forLoopLinkageDuplicate';
    }

    linkedReturnByForId.set(edge.source.node_id, edge.destination.node_id);
    linkedForByReturnId.set(edge.destination.node_id, edge.source.node_id);
  }

  for (const node of nodes) {
    if (
      (node.type === 'for' && !linkedReturnByForId.has(node.id)) ||
      (node.type === 'for_return' && !linkedForByReturnId.has(node.id))
    ) {
      return 'nodes.forLoopLinkageMissing';
    }
  }

  const hasPath = (startId: string, targetId: string): boolean =>
    startId === targetId || walk([startId], outgoing).has(targetId);

  const supportsNestedIterateBody = (
    bodyPathNodeIds: Set<string>,
    iterateNodeIds: string[],
    collectNodeIds: string[],
    returnId: string,
    forId: string
  ): boolean => {
    if (iterateNodeIds.length !== 1 || collectNodeIds.length !== 1) {
      return false;
    }

    const iterateId = iterateNodeIds[0];
    const collectId = collectNodeIds[0];
    if (iterateId === undefined || collectId === undefined || !hasPath(iterateId, collectId)) {
      return false;
    }

    const iterateCollectionEdges = edges.filter(
      (edge) => edge.destination.node_id === iterateId && edge.destination.field === 'collection'
    );
    const iterateCollectionSourceId = iterateCollectionEdges[0]?.source.node_id;
    if (
      iterateCollectionEdges.length !== 1 ||
      iterateCollectionSourceId === undefined ||
      (iterateCollectionSourceId !== forId && !bodyPathNodeIds.has(iterateCollectionSourceId))
    ) {
      return false;
    }

    const returnOutputEdges = edges.filter(
      (edge) => edge.destination.node_id === returnId && edge.destination.field === 'output'
    );
    if (
      returnOutputEdges.length !== 1 ||
      returnOutputEdges[0]?.source.node_id !== collectId ||
      returnOutputEdges[0]?.source.field !== 'collection'
    ) {
      return false;
    }

    const unsupportedReturnInput = edges.some(
      (edge) =>
        edge.destination.node_id === returnId &&
        edge.destination.field !== 'output' &&
        edge.destination.field !== 'continue_condition' &&
        (edge.destination.field !== 'state' || edge.source.node_id !== forId || edge.source.field !== 'state')
    );
    if (unsupportedReturnInput) {
      return false;
    }

    const collectCollectionEdges = edges.filter(
      (edge) => edge.destination.node_id === collectId && edge.destination.field === 'collection'
    );
    const collectItemEdges = edges.filter(
      (edge) => edge.destination.node_id === collectId && edge.destination.field === 'item'
    );
    if (collectCollectionEdges.length !== 0 || collectItemEdges.length !== 1) {
      return false;
    }

    const collectItemSourceId = collectItemEdges[0]?.source.node_id;
    if (collectItemSourceId === undefined || !hasPath(iterateId, collectItemSourceId)) {
      return false;
    }

    for (const bodyNodeId of bodyPathNodeIds) {
      if (bodyNodeId === iterateId || bodyNodeId === collectId || bodyNodeId === returnId) {
        continue;
      }
      if (!hasPath(bodyNodeId, collectId)) {
        return false;
      }
      if (!hasPath(bodyNodeId, iterateId) && !hasPath(iterateId, bodyNodeId)) {
        return false;
      }
    }

    return true;
  };

  const getSupportedNestedForBody = (
    forId: string,
    reachableBodyNodeIds: Set<string>,
    reachableReturnIds: string[]
  ): { bodyPathNodeIds: Set<string>; returnId: string } | null => {
    const outerReturnId = linkedReturnByForId.get(forId);
    if (outerReturnId === undefined || !reachableReturnIds.includes(outerReturnId)) {
      return null;
    }

    const innerForIds = [...reachableBodyNodeIds].filter((nodeId) => nodesById.get(nodeId)?.type === 'for');
    const directInnerForIds = innerForIds.filter(
      (innerForId) =>
        !innerForIds.some((otherInnerForId) => otherInnerForId !== innerForId && hasPath(otherInnerForId, innerForId))
    );
    if (directInnerForIds.length === 0) {
      return null;
    }

    const innerBodyPathNodeIds = new Set<string>();

    for (const innerForId of directInnerForIds) {
      const innerIterationEdges = edges.filter(
        (edge) => edge.source.node_id === innerForId && ITERATION_OUTPUT_FIELDS.has(edge.source.field)
      );
      if (innerIterationEdges.length === 0) {
        return null;
      }

      const innerReachableBodyNodeIds = walk(
        innerIterationEdges.map((edge) => edge.destination.node_id),
        outgoing
      );
      const innerReachableReturnIds = [...innerReachableBodyNodeIds].filter(
        (nodeId) => nodesById.get(nodeId)?.type === 'for_return'
      );
      const innerReturnId = linkedReturnByForId.get(innerForId);
      if (innerReturnId === undefined || !innerReachableReturnIds.includes(innerReturnId)) {
        return null;
      }

      const innerReturnAncestors = walk([innerReturnId], incoming);
      const childBodyPathNodeIds = new Set(
        [...innerReachableBodyNodeIds].filter((nodeId) => nodeId === innerReturnId || innerReturnAncestors.has(nodeId))
      );
      childBodyPathNodeIds.add(innerReturnId);

      const innerNestedForIds = [...childBodyPathNodeIds].filter((nodeId) => nodesById.get(nodeId)?.type === 'for');
      if ([...childBodyPathNodeIds].some((nodeId) => nodesById.get(nodeId)?.type === 'iterate')) {
        return null;
      }

      const innerNestedBody =
        innerNestedForIds.length > 0
          ? getSupportedNestedForBody(innerForId, innerReachableBodyNodeIds, innerReachableReturnIds)
          : null;
      if (innerNestedForIds.length > 0 && innerNestedBody === null) {
        return null;
      }
      if (innerNestedBody !== null) {
        for (const bodyNodeId of innerNestedBody.bodyPathNodeIds) {
          childBodyPathNodeIds.add(bodyNodeId);
        }
      }

      const innerCollectionEdges = edges.filter(
        (edge) => edge.destination.node_id === innerForId && edge.destination.field === 'collection'
      );
      const innerCollectionSourceId = innerCollectionEdges[0]?.source.node_id;
      if (
        innerCollectionEdges.length !== 1 ||
        innerCollectionSourceId === undefined ||
        (innerCollectionSourceId !== forId && !reachableBodyNodeIds.has(innerCollectionSourceId))
      ) {
        return null;
      }

      const unsupportedInnerReturnInput = edges.some(
        (edge) =>
          edge.destination.node_id === innerReturnId &&
          edge.destination.field === 'state' &&
          edge.source.node_id !== innerForId &&
          !childBodyPathNodeIds.has(edge.source.node_id)
      );
      if (unsupportedInnerReturnInput) {
        return null;
      }

      for (const bodyNodeId of childBodyPathNodeIds) {
        innerBodyPathNodeIds.add(bodyNodeId);
      }
    }

    if (
      new Set(reachableReturnIds.filter((returnId) => !innerBodyPathNodeIds.has(returnId))).size !== 1 ||
      !reachableReturnIds.includes(outerReturnId)
    ) {
      return null;
    }

    const outerReturnOutputEdges = edges.filter(
      (edge) => edge.destination.node_id === outerReturnId && edge.destination.field === 'output'
    );
    if (outerReturnOutputEdges.length !== 1) {
      return null;
    }

    const unsupportedOuterReturnInput = edges.some(
      (edge) =>
        edge.destination.node_id === outerReturnId &&
        edge.destination.field !== 'output' &&
        edge.destination.field !== 'continue_condition' &&
        (edge.destination.field !== 'state' || edge.source.node_id !== forId || edge.source.field !== 'state')
    );
    if (unsupportedOuterReturnInput) {
      return null;
    }

    const outerPreparationNodeIds = new Set<string>();
    for (const innerForId of directInnerForIds) {
      for (const bodyNodeId of reachableBodyNodeIds) {
        if (walk([innerForId], incoming).has(bodyNodeId)) {
          outerPreparationNodeIds.add(bodyNodeId);
        }
      }
      outerPreparationNodeIds.add(innerForId);
    }

    const innerFinalDescendantNodeIds = new Set<string>();
    for (const innerForId of directInnerForIds) {
      for (const destinationId of edges
        .filter((edge) => edge.source.node_id === innerForId && edge.source.field === 'output_collection')
        .map((edge) => edge.destination.node_id)) {
        for (const descendantId of walk([destinationId], outgoing)) {
          innerFinalDescendantNodeIds.add(descendantId);
        }
      }
    }

    const continuationNodeIds = new Set(
      [...reachableBodyNodeIds].filter(
        (nodeId) =>
          !outerPreparationNodeIds.has(nodeId) && !innerBodyPathNodeIds.has(nodeId) && nodeId !== outerReturnId
      )
    );
    if (
      edges.some(
        (edge) =>
          edge.destination.node_id === outerReturnId &&
          edge.destination.field === 'continue_condition' &&
          edge.source.node_id !== forId &&
          !continuationNodeIds.has(edge.source.node_id) &&
          !(directInnerForIds.includes(edge.source.node_id) && FINAL_OUTPUT_FIELDS.has(edge.source.field))
      )
    ) {
      return null;
    }
    if ([...continuationNodeIds].some((nodeId) => !innerFinalDescendantNodeIds.has(nodeId))) {
      return null;
    }
    if ([...continuationNodeIds].some((nodeId) => !hasPath(nodeId, outerReturnId))) {
      return null;
    }
    if (
      [...continuationNodeIds].some((nodeId) => {
        const type = nodesById.get(nodeId)?.type;
        return type === 'for' || type === 'iterate' || type === 'for_return';
      })
    ) {
      return null;
    }
    if (
      [...continuationNodeIds].some((nodeId) =>
        edges.some(
          (edge) =>
            edge.destination.node_id === nodeId &&
            (innerBodyPathNodeIds.has(edge.source.node_id) ||
              (directInnerForIds.includes(edge.source.node_id) && edge.source.field !== 'output_collection'))
        )
      )
    ) {
      return null;
    }

    const outerReturnOutputSource = outerReturnOutputEdges[0]?.source;
    if (outerReturnOutputSource !== undefined && directInnerForIds.includes(outerReturnOutputSource.node_id)) {
      if (
        directInnerForIds.length !== 1 ||
        outerReturnOutputSource.field !== 'output_collection' ||
        continuationNodeIds.size > 0
      ) {
        return null;
      }
    } else if (outerReturnOutputSource === undefined || !continuationNodeIds.has(outerReturnOutputSource.node_id)) {
      return null;
    }

    if (
      directInnerForIds.some(
        (innerForId) =>
          !edges
            .filter((edge) => edge.source.node_id === innerForId && FINAL_OUTPUT_FIELDS.has(edge.source.field))
            .some(
              (edge) => continuationNodeIds.has(edge.destination.node_id) || edge.destination.node_id === outerReturnId
            )
      )
    ) {
      return null;
    }

    const bodyPathNodeIds = new Set([
      ...outerPreparationNodeIds,
      ...innerBodyPathNodeIds,
      ...continuationNodeIds,
      outerReturnId,
    ]);
    if ([...reachableBodyNodeIds].some((nodeId) => !bodyPathNodeIds.has(nodeId))) {
      return null;
    }
    if (
      [...outerPreparationNodeIds].some(
        (nodeId) =>
          !directInnerForIds.includes(nodeId) &&
          (nodesById.get(nodeId)?.type === 'for' || nodesById.get(nodeId)?.type === 'iterate')
      )
    ) {
      return null;
    }
    for (const bodyNodeId of outerPreparationNodeIds) {
      if (directInnerForIds.includes(bodyNodeId) || innerBodyPathNodeIds.has(bodyNodeId)) {
        continue;
      }
      if (!directInnerForIds.some((innerForId) => hasPath(bodyNodeId, innerForId))) {
        return null;
      }
    }

    return { bodyPathNodeIds, returnId: outerReturnId };
  };

  const matchingForIdsByReturnId = new Map<string, string[]>();

  for (const node of nodes) {
    if (node.type !== 'for') {
      continue;
    }

    const collectionInputs = edges.filter(
      (edge) => edge.destination.node_id === node.id && edge.destination.field === 'collection'
    );
    const stateInputs = edges.filter(
      (edge) => edge.destination.node_id === node.id && edge.destination.field === 'state'
    );

    if (collectionInputs.length > 1 || stateInputs.length > 1) {
      return 'nodes.forLoopInputCount';
    }

    const iterationEdges = edges.filter(
      (edge) => edge.source.node_id === node.id && ITERATION_OUTPUT_FIELDS.has(edge.source.field)
    );

    if (iterationEdges.length === 0) {
      return 'nodes.forLoopMissingIterationOutput';
    }

    const reachableBodyNodeIds = walk(
      iterationEdges.map((edge) => edge.destination.node_id),
      outgoing
    );
    const reachableReturnIds = [...reachableBodyNodeIds].filter(
      (nodeId) => nodesById.get(nodeId)?.type === 'for_return'
    );
    const linkedReturnId = linkedReturnByForId.get(node.id);

    if (reachableBodyNodeIds.has(node.id)) {
      return 'nodes.forLoopBodyEscape';
    }

    const nestedForNodeIds = [...reachableBodyNodeIds].filter(
      (nodeId) =>
        nodeId !== node.id &&
        nodesById.get(nodeId)?.type === 'for' &&
        ![...reachableBodyNodeIds].some(
          (otherNodeId) =>
            otherNodeId !== nodeId && nodesById.get(otherNodeId)?.type === 'for' && hasPath(otherNodeId, nodeId)
        )
    );
    const nestedBody =
      nestedForNodeIds.length > 0 ? getSupportedNestedForBody(node.id, reachableBodyNodeIds, reachableReturnIds) : null;
    if (nestedForNodeIds.length > 0 && nestedBody === null) {
      return 'nodes.forLoopNestedUnsupported';
    }

    if (nestedBody === null && (linkedReturnId === undefined || !reachableReturnIds.includes(linkedReturnId))) {
      return 'nodes.forLoopReturnCount';
    }

    const returnId = nestedBody?.returnId ?? linkedReturnId;
    if (returnId === undefined) {
      return 'nodes.forLoopReturnCount';
    }
    matchingForIdsByReturnId.set(returnId, [...(matchingForIdsByReturnId.get(returnId) ?? []), node.id]);

    const bodyPathNodeIds =
      nestedBody?.bodyPathNodeIds ??
      (() => {
        const returnAncestorIds = walk(incoming.get(returnId) ?? [], incoming);
        const path = new Set(
          [...reachableBodyNodeIds].filter((nodeId) => nodeId === returnId || returnAncestorIds.has(nodeId))
        );
        path.add(returnId);
        return path;
      })();

    if ([...reachableBodyNodeIds].some((nodeId) => !bodyPathNodeIds.has(nodeId))) {
      return 'nodes.forLoopUnterminatedBody';
    }

    const iterateNodeIds = [...bodyPathNodeIds].filter((nodeId) => nodesById.get(nodeId)?.type === 'iterate');
    if (
      iterateNodeIds.length > 0 &&
      !supportsNestedIterateBody(
        bodyPathNodeIds,
        iterateNodeIds,
        [...bodyPathNodeIds].filter((nodeId) => nodesById.get(nodeId)?.type === 'collect'),
        returnId,
        node.id
      )
    ) {
      return 'nodes.forLoopIterateUnsupported';
    }

    for (const bodyNodeId of bodyPathNodeIds) {
      for (const sourceId of incoming.get(bodyNodeId) ?? []) {
        if (sourceId === node.id || bodyPathNodeIds.has(sourceId)) {
          continue;
        }

        if ([...walk([sourceId], incoming)].some((sourceNodeId) => nodesById.get(sourceNodeId)?.type === 'iterate')) {
          return 'nodes.forLoopIteratorInputUnsupported';
        }
      }
    }

    if (
      edges.some(
        (edge) =>
          edge.source.node_id === node.id &&
          FINAL_OUTPUT_FIELDS.has(edge.source.field) &&
          (bodyPathNodeIds.has(edge.destination.node_id) || hasPath(edge.destination.node_id, returnId))
      )
    ) {
      return 'nodes.forLoopFinalOutputInBody';
    }

    for (const bodyNodeId of bodyPathNodeIds) {
      if (bodyNodeId === returnId) {
        continue;
      }

      if ((outgoing.get(bodyNodeId) ?? []).some((destinationId) => !bodyPathNodeIds.has(destinationId))) {
        return 'nodes.forLoopBodyEscape';
      }
    }
  }

  for (const node of nodes) {
    if (node.type !== 'for_return') {
      continue;
    }

    if (matchingForIdsByReturnId.get(node.id)?.length !== 1) {
      return 'nodes.forReturnOwnership';
    }

    if (
      edges.filter((edge) => edge.destination.node_id === node.id && edge.destination.field === 'output').length > 1 ||
      edges.filter((edge) => edge.destination.node_id === node.id && edge.destination.field === 'state').length > 1 ||
      edges.filter((edge) => edge.destination.node_id === node.id && edge.destination.field === 'continue_condition')
        .length > 1
    ) {
      return 'nodes.forReturnInputCount';
    }
  }

  return null;
};
