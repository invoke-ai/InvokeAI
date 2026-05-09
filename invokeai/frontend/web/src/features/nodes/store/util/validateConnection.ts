import type { Connection as NullableConnection } from '@xyflow/react';
import type { Templates } from 'features/nodes/store/types';
import { areTypesEqual } from 'features/nodes/store/util/areTypesEqual';
import {
  CONNECTOR_INPUT_HANDLE,
  CONNECTOR_OUTPUT_HANDLE,
  resolveConnectorSource,
} from 'features/nodes/store/util/connectorTopology';
import { getCollectItemType } from 'features/nodes/store/util/getCollectItemType';
import { getHasCycles } from 'features/nodes/store/util/getHasCycles';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { FieldType } from 'features/nodes/types/field';
import type { AnyEdge, AnyNode, InvocationNode } from 'features/nodes/types/invocation';
import { isConnectorNode, isInvocationNode } from 'features/nodes/types/invocation';
import type { SetNonNullable } from 'type-fest';

type Connection = SetNonNullable<NullableConnection>;

type ValidateConnectionFunc = (
  connection: Connection,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates,
  ignoreEdge: AnyEdge | null,
  strict?: boolean
) => string | null;

type EffectiveSource = {
  node: InvocationNode;
  handle: string;
  fieldTemplate: NonNullable<Templates[string]>['outputs'][string];
};

const getEqualityPredicate =
  (c: Connection) =>
  (e: AnyEdge): boolean => {
    return (
      e.target === c.target &&
      e.targetHandle === c.targetHandle &&
      e.source === c.source &&
      e.sourceHandle === c.sourceHandle
    );
  };

const getTargetEqualityPredicate =
  (c: Connection) =>
  (e: AnyEdge): boolean => {
    return e.target === c.target && e.targetHandle === c.targetHandle;
  };

const IF_INPUT_HANDLES = ['true_input', 'false_input'] as const;

const isIfInputHandle = (handle: string): handle is (typeof IF_INPUT_HANDLES)[number] => {
  return IF_INPUT_HANDLES.includes(handle as (typeof IF_INPUT_HANDLES)[number]);
};

const isSingleCollectionPairOfSameBaseType = (firstType: FieldType, secondType: FieldType) => {
  const isSingleToCollection =
    firstType.cardinality === 'SINGLE' && secondType.cardinality === 'COLLECTION' && firstType.name === secondType.name;
  const isCollectionToSingle =
    firstType.cardinality === 'COLLECTION' && secondType.cardinality === 'SINGLE' && firstType.name === secondType.name;
  return firstType.batch === secondType.batch && (isSingleToCollection || isCollectionToSingle);
};

const areFieldTypesCompatible = (firstType: FieldType, secondType: FieldType) =>
  validateConnectionTypes(firstType, secondType) ||
  validateConnectionTypes(secondType, firstType) ||
  isSingleCollectionPairOfSameBaseType(firstType, secondType);

type ConnectorTerminalTargetEdge = AnyEdge & {
  type: 'default';
  sourceHandle: string;
  targetHandle: string;
};

const getConnectorTerminalTargetEdges = (connectorId: string, nodes: AnyNode[], edges: AnyEdge[]) => {
  const visited = new Set<string>();
  const resolve = (currentConnectorId: string): ConnectorTerminalTargetEdge[] => {
    if (visited.has(currentConnectorId)) {
      return [];
    }
    visited.add(currentConnectorId);

    return edges.flatMap((edge) => {
      if (
        edge.type !== 'default' ||
        edge.source !== currentConnectorId ||
        edge.sourceHandle !== CONNECTOR_OUTPUT_HANDLE ||
        typeof edge.targetHandle !== 'string'
      ) {
        return [];
      }

      const targetNode = nodes.find((node) => node.id === edge.target);
      if (targetNode && isConnectorNode(targetNode)) {
        return resolve(targetNode.id);
      }

      return [edge as ConnectorTerminalTargetEdge];
    });
  };

  return resolve(connectorId);
};

const getConnectorSubgraphEdgeIds = (connectorId: string, nodes: AnyNode[], edges: AnyEdge[]) => {
  const visited = new Set<string>();
  const edgeIds = new Set<string>();

  const visit = (currentConnectorId: string) => {
    if (visited.has(currentConnectorId)) {
      return;
    }
    visited.add(currentConnectorId);

    edges.forEach((edge) => {
      if (
        edge.type !== 'default' ||
        edge.source !== currentConnectorId ||
        edge.sourceHandle !== CONNECTOR_OUTPUT_HANDLE ||
        typeof edge.targetHandle !== 'string'
      ) {
        return;
      }

      edgeIds.add(edge.id);

      const targetNode = nodes.find((node) => node.id === edge.target);
      if (targetNode && isConnectorNode(targetNode)) {
        visit(targetNode.id);
      }
    });
  };

  visit(connectorId);
  return edgeIds;
};

const getEffectiveSource = (
  sourceId: string,
  sourceHandle: string,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates
): EffectiveSource | 'nodes.missingNode' | 'nodes.missingInvocationTemplate' | 'nodes.missingFieldTemplate' | null => {
  const sourceNode = nodes.find((n) => n.id === sourceId);
  if (!sourceNode) {
    return 'nodes.missingNode';
  }

  if (isConnectorNode(sourceNode)) {
    if (sourceHandle !== CONNECTOR_OUTPUT_HANDLE) {
      return 'nodes.missingFieldTemplate';
    }

    const resolvedSource = resolveConnectorSource(sourceNode.id, nodes, edges);
    if (!resolvedSource) {
      return null;
    }

    return getEffectiveSource(resolvedSource.nodeId, resolvedSource.fieldName, nodes, edges, templates);
  }

  if (!isInvocationNode(sourceNode)) {
    return 'nodes.missingInvocationTemplate';
  }

  const sourceTemplate = templates[sourceNode.data.type];
  if (!sourceTemplate) {
    return 'nodes.missingInvocationTemplate';
  }

  const sourceFieldTemplate = sourceTemplate.outputs[sourceHandle];
  if (!sourceFieldTemplate) {
    return 'nodes.missingFieldTemplate';
  }

  return {
    node: sourceNode,
    handle: sourceHandle,
    fieldTemplate: sourceFieldTemplate,
  };
};

const getEffectiveSourceForEdge = (
  edge: AnyEdge,
  nodes: AnyNode[],
  edges: AnyEdge[],
  templates: Templates
): EffectiveSource | 'nodes.missingNode' | 'nodes.missingInvocationTemplate' | 'nodes.missingFieldTemplate' | null => {
  if (edge.type !== 'default' || typeof edge.sourceHandle !== 'string') {
    return null;
  }

  return getEffectiveSource(edge.source, edge.sourceHandle, nodes, edges, templates);
};

/**
 * Validates a connection between two fields
 * @returns A translation key for an error if the connection is invalid, otherwise null
 */
export const validateConnection: ValidateConnectionFunc = (
  c,
  nodes,
  edges,
  templates,
  ignoreEdge,
  strict = true
): string | null => {
  if (c.source === c.target) {
    return 'nodes.cannotConnectToSelf';
  }

  if (strict) {
    /**
     * We may need to ignore an edge when validating a connection.
     *
     * For example, while an edge is being updated, it still exists in the array of edges. As we validate the new connection,
     * the user experience should be that the edge is temporarily removed from the graph, so we need to ignore it, else
     * the validation will fail unexpectedly.
     */
    const filteredEdges = edges.filter((e) => e.id !== ignoreEdge?.id);

    if (filteredEdges.some(getEqualityPredicate(c))) {
      // We already have a connection from this source to this target
      return 'nodes.cannotDuplicateConnection';
    }

    const targetNode = nodes.find((n) => n.id === c.target);
    const sourceNode = nodes.find((n) => n.id === c.source);
    if (!targetNode) {
      return 'nodes.missingNode';
    }

    const effectiveSource = getEffectiveSource(c.source, c.sourceHandle, nodes, filteredEdges, templates);
    if (effectiveSource === 'nodes.missingNode') {
      return 'nodes.missingNode';
    }
    if (effectiveSource === 'nodes.missingInvocationTemplate') {
      return 'nodes.missingInvocationTemplate';
    }
    if (effectiveSource === 'nodes.missingFieldTemplate') {
      return 'nodes.missingFieldTemplate';
    }

    if (isConnectorNode(targetNode)) {
      if (c.targetHandle !== CONNECTOR_INPUT_HANDLE) {
        return 'nodes.missingFieldTemplate';
      }

      if (filteredEdges.find(getTargetEqualityPredicate(c))) {
        return 'nodes.inputMayOnlyHaveOneConnection';
      }

      if (effectiveSource) {
        const connectorSubgraphEdgeIds = getConnectorSubgraphEdgeIds(targetNode.id, nodes, filteredEdges);
        const stagedEdges = filteredEdges.filter((edge) => !connectorSubgraphEdgeIds.has(edge.id));
        const terminalTargetEdges = getConnectorTerminalTargetEdges(targetNode.id, nodes, filteredEdges);

        for (const terminalTargetEdge of terminalTargetEdges) {
          const downstreamValidation = validateConnection(
            {
              source: c.source,
              sourceHandle: c.sourceHandle,
              target: terminalTargetEdge.target,
              targetHandle: terminalTargetEdge.targetHandle,
            },
            nodes,
            stagedEdges,
            templates,
            null,
            true
          );

          if (downstreamValidation !== null) {
            return downstreamValidation;
          }
        }
      }

      // Unresolved connector chains are allowed to terminate on another connector.
      return null;
    }

    if (!isInvocationNode(targetNode)) {
      return 'nodes.missingInvocationTemplate';
    }

    const targetTemplate = templates[targetNode.data.type];
    if (!targetTemplate) {
      return 'nodes.missingInvocationTemplate';
    }

    const targetFieldTemplate = targetTemplate.inputs[c.targetHandle];
    if (!targetFieldTemplate) {
      return 'nodes.missingFieldTemplate';
    }

    if (targetFieldTemplate.input === 'direct') {
      return 'nodes.cannotConnectToDirectInput';
    }

    if (!effectiveSource) {
      if (sourceNode && isConnectorNode(sourceNode) && c.sourceHandle === CONNECTOR_OUTPUT_HANDLE) {
        const existingTerminalTargetEdges = getConnectorTerminalTargetEdges(sourceNode.id, nodes, filteredEdges).filter(
          (edge) => !(edge.target === c.target && edge.targetHandle === c.targetHandle)
        );

        for (const terminalTargetEdge of existingTerminalTargetEdges) {
          const constrainedTargetNode = nodes.find((node) => node.id === terminalTargetEdge.target);
          if (!constrainedTargetNode || !isInvocationNode(constrainedTargetNode)) {
            return 'nodes.missingInvocationTemplate';
          }

          const constrainedTargetTemplate = templates[constrainedTargetNode.data.type];
          if (!constrainedTargetTemplate) {
            return 'nodes.missingInvocationTemplate';
          }

          const constrainedTargetFieldTemplate = constrainedTargetTemplate.inputs[terminalTargetEdge.targetHandle];
          if (!constrainedTargetFieldTemplate) {
            return 'nodes.missingFieldTemplate';
          }

          if (!areFieldTypesCompatible(constrainedTargetFieldTemplate.type, targetFieldTemplate.type)) {
            return 'nodes.fieldTypesMustMatch';
          }
        }

        return null;
      }

      return 'nodes.fieldTypesMustMatch';
    }

    const { node: resolvedSourceNode, handle: sourceHandle, fieldTemplate: sourceFieldTemplate } = effectiveSource;

    if (targetNode.data.type === 'collect' && c.targetHandle === 'item') {
      // Collect nodes shouldn't mix and match field types.
      const collectItemType = getCollectItemType(templates, nodes, filteredEdges, targetNode.id);
      if (collectItemType && !areTypesEqual(sourceFieldTemplate.type, collectItemType)) {
        return 'nodes.cannotMixAndMatchCollectionItemTypes';
      }
    }

    if (
      resolvedSourceNode.data.type === 'collect' &&
      sourceHandle === 'collection' &&
      targetNode.data.type === 'collect' &&
      c.targetHandle === 'collection'
    ) {
      // Chained collect nodes should preserve a single item type when both ends are already typed.
      const sourceCollectItemType = getCollectItemType(templates, nodes, filteredEdges, resolvedSourceNode.id);
      const targetCollectItemType = getCollectItemType(templates, nodes, filteredEdges, targetNode.id);
      if (
        sourceCollectItemType &&
        targetCollectItemType &&
        !areTypesEqual(sourceCollectItemType, targetCollectItemType)
      ) {
        return 'nodes.cannotMixAndMatchCollectionItemTypes';
      }
    }

    if (targetNode.data.type === 'if' && isIfInputHandle(c.targetHandle)) {
      const siblingHandle = c.targetHandle === 'true_input' ? 'false_input' : 'true_input';
      const siblingInputEdge = filteredEdges.find((e) => e.target === c.target && e.targetHandle === siblingHandle);

      if (siblingInputEdge) {
        const siblingEffectiveSource = getEffectiveSourceForEdge(siblingInputEdge, nodes, filteredEdges, templates);
        if (siblingEffectiveSource === 'nodes.missingNode') {
          return 'nodes.missingNode';
        }
        if (siblingEffectiveSource === 'nodes.missingInvocationTemplate') {
          return 'nodes.missingInvocationTemplate';
        }
        if (siblingEffectiveSource === 'nodes.missingFieldTemplate') {
          return 'nodes.missingFieldTemplate';
        }
        if (!siblingEffectiveSource) {
          return 'nodes.fieldTypesMustMatch';
        }

        const areIfInputTypesCompatible =
          validateConnectionTypes(sourceFieldTemplate.type, siblingEffectiveSource.fieldTemplate.type) ||
          validateConnectionTypes(siblingEffectiveSource.fieldTemplate.type, sourceFieldTemplate.type) ||
          isSingleCollectionPairOfSameBaseType(sourceFieldTemplate.type, siblingEffectiveSource.fieldTemplate.type);

        if (!areIfInputTypesCompatible) {
          return 'nodes.fieldTypesMustMatch';
        }
      }
    }

    if (filteredEdges.find(getTargetEqualityPredicate(c))) {
      // CollectionItemField inputs can have multiple input connections
      if (targetFieldTemplate.type.name !== 'CollectionItemField') {
        return 'nodes.inputMayOnlyHaveOneConnection';
      }
    }

    if (resolvedSourceNode.data.type === 'if' && sourceHandle === 'value') {
      const ifInputEdges = filteredEdges.filter(
        (e) =>
          e.target === resolvedSourceNode.id && typeof e.targetHandle === 'string' && isIfInputHandle(e.targetHandle)
      );
      const ifInputTypes = ifInputEdges.flatMap((edge) => {
        const ifInputSource = getEffectiveSourceForEdge(edge, nodes, filteredEdges, templates);
        if (!ifInputSource || typeof ifInputSource === 'string') {
          return [];
        }
        return [ifInputSource.fieldTemplate.type];
      });

      if (ifInputTypes.length > 0) {
        const areAllIfInputsCompatibleWithTarget = ifInputTypes.every((ifInputType) =>
          validateConnectionTypes(ifInputType, targetFieldTemplate.type)
        );
        if (!areAllIfInputsCompatibleWithTarget) {
          return 'nodes.fieldTypesMustMatch';
        }
      } else if (!validateConnectionTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
        return 'nodes.fieldTypesMustMatch';
      }
    } else if (!validateConnectionTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
      return 'nodes.fieldTypesMustMatch';
    }
  }

  if (getHasCycles(c.source, c.target, nodes, edges)) {
    return 'nodes.connectionWouldCreateCycle';
  }

  return null;
};
