import type { Connection as NullableConnection } from '@xyflow/react';
import type { Templates } from 'features/nodes/store/types';
import { areTypesEqual } from 'features/nodes/store/util/areTypesEqual';
import { getCollectItemType } from 'features/nodes/store/util/getCollectItemType';
import { getHasCycles } from 'features/nodes/store/util/getHasCycles';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { AnyEdge, AnyNode } from 'features/nodes/types/invocation';
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

    const sourceNode = nodes.find((n) => n.id === c.source);
    if (!sourceNode) {
      return 'nodes.missingNode';
    }

    const targetNode = nodes.find((n) => n.id === c.target);
    if (!targetNode) {
      return 'nodes.missingNode';
    }

    const sourceTemplate = templates[sourceNode.data.type];
    if (!sourceTemplate) {
      return 'nodes.missingInvocationTemplate';
    }

    const targetTemplate = templates[targetNode.data.type];
    if (!targetTemplate) {
      return 'nodes.missingInvocationTemplate';
    }

    const sourceFieldTemplate = sourceTemplate.outputs[c.sourceHandle];
    if (!sourceFieldTemplate) {
      return 'nodes.missingFieldTemplate';
    }

    const targetFieldTemplate = targetTemplate.inputs[c.targetHandle];
    if (!targetFieldTemplate) {
      return 'nodes.missingFieldTemplate';
    }

    if (targetFieldTemplate.input === 'direct') {
      return 'nodes.cannotConnectToDirectInput';
    }

    if (targetNode.data.type === 'collect' && c.targetHandle === 'item') {
      // Collect nodes shouldn't mix and match field types.
      const collectItemType = getCollectItemType(templates, nodes, edges, targetNode.id);
      if (collectItemType && !areTypesEqual(sourceFieldTemplate.type, collectItemType)) {
        return 'nodes.cannotMixAndMatchCollectionItemTypes';
      }
    }

    if (filteredEdges.find(getTargetEqualityPredicate(c))) {
      // CollectionItemField inputs can have multiple input connections
      if (targetFieldTemplate.type.name !== 'CollectionItemField') {
        return 'nodes.inputMayOnlyHaveOneConnection';
      }
    }

    if (!validateConnectionTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
      return 'nodes.fieldTypesMustMatch';
    }
  }

  if (getHasCycles(c.source, c.target, nodes, edges)) {
    return 'nodes.connectionWouldCreateCycle';
  }

  return null;
};
