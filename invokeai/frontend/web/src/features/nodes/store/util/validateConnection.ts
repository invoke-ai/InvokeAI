import type { Templates } from 'features/nodes/store/types';
import { areTypesEqual } from 'features/nodes/store/util/areTypesEqual';
import { getCollectItemType } from 'features/nodes/store/util/getCollectItemType';
import { getHasCycles } from 'features/nodes/store/util/getHasCycles';
import { validateConnectionTypes } from 'features/nodes/store/util/validateConnectionTypes';
import type { AnyNode } from 'features/nodes/types/invocation';
import type { Connection as NullableConnection, Edge } from 'reactflow';
import type { O } from 'ts-toolbelt';

type Connection = O.NonNullable<NullableConnection>;

export type ValidationResult =
  | {
      isValid: true;
      messageTKey?: string;
    }
  | {
      isValid: false;
      messageTKey: string;
    };

type ValidateConnectionFunc = (
  connection: Connection,
  nodes: AnyNode[],
  edges: Edge[],
  templates: Templates,
  ignoreEdge: Edge | null,
  strict?: boolean
) => ValidationResult;

const getEqualityPredicate =
  (c: Connection) =>
  (e: Edge): boolean => {
    return (
      e.target === c.target &&
      e.targetHandle === c.targetHandle &&
      e.source === c.source &&
      e.sourceHandle === c.sourceHandle
    );
  };

const getTargetEqualityPredicate =
  (c: Connection) =>
  (e: Edge): boolean => {
    return e.target === c.target && e.targetHandle === c.targetHandle;
  };

export const buildAcceptResult = (): ValidationResult => ({ isValid: true });
export const buildRejectResult = (messageTKey: string): ValidationResult => ({ isValid: false, messageTKey });

export const validateConnection: ValidateConnectionFunc = (c, nodes, edges, templates, ignoreEdge, strict = true) => {
  if (c.source === c.target) {
    return buildRejectResult('nodes.cannotConnectToSelf');
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
      return buildRejectResult('nodes.cannotDuplicateConnection');
    }

    const sourceNode = nodes.find((n) => n.id === c.source);
    if (!sourceNode) {
      return buildRejectResult('nodes.missingNode');
    }

    const targetNode = nodes.find((n) => n.id === c.target);
    if (!targetNode) {
      return buildRejectResult('nodes.missingNode');
    }

    const sourceTemplate = templates[sourceNode.data.type];
    if (!sourceTemplate) {
      return buildRejectResult('nodes.missingInvocationTemplate');
    }

    const targetTemplate = templates[targetNode.data.type];
    if (!targetTemplate) {
      return buildRejectResult('nodes.missingInvocationTemplate');
    }

    const sourceFieldTemplate = sourceTemplate.outputs[c.sourceHandle];
    if (!sourceFieldTemplate) {
      return buildRejectResult('nodes.missingFieldTemplate');
    }

    const targetFieldTemplate = targetTemplate.inputs[c.targetHandle];
    if (!targetFieldTemplate) {
      return buildRejectResult('nodes.missingFieldTemplate');
    }

    if (targetFieldTemplate.input === 'direct') {
      return buildRejectResult('nodes.cannotConnectToDirectInput');
    }

    if (targetNode.data.type === 'collect' && c.targetHandle === 'item') {
      // Collect nodes shouldn't mix and match field types.
      const collectItemType = getCollectItemType(templates, nodes, edges, targetNode.id);
      if (collectItemType && !areTypesEqual(sourceFieldTemplate.type, collectItemType)) {
        return buildRejectResult('nodes.cannotMixAndMatchCollectionItemTypes');
      }
    }

    if (filteredEdges.find(getTargetEqualityPredicate(c))) {
      // CollectionItemField inputs can have multiple input connections
      if (targetFieldTemplate.type.name !== 'CollectionItemField') {
        return buildRejectResult('nodes.inputMayOnlyHaveOneConnection');
      }
    }

    if (!validateConnectionTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
      return buildRejectResult('nodes.fieldTypesMustMatch');
    }
  }

  if (getHasCycles(c.source, c.target, nodes, edges)) {
    return buildRejectResult('nodes.connectionWouldCreateCycle');
  }

  return buildAcceptResult();
};
