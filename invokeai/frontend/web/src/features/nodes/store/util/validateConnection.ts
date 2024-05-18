import type { Templates } from 'features/nodes/store/types';
import { areTypesEqual } from 'features/nodes/store/util/areTypesEqual';
import { getCollectItemType } from 'features/nodes/store/util/getCollectItemType';
import type { AnyNode } from 'features/nodes/types/invocation';
import type { Connection as NullableConnection, Edge } from 'reactflow';
import type { O } from 'ts-toolbelt';

type Connection = O.NonNullable<NullableConnection>;

export type ValidateConnectionResult = {
  isValid: boolean;
  messageTKey?: string;
};

export type ValidateConnectionFunc = (
  connection: Connection,
  nodes: AnyNode[],
  edges: Edge[],
  templates: Templates,
  ignoreEdge: Edge | null
) => ValidateConnectionResult;

export const buildResult = (isValid: boolean, messageTKey?: string): ValidateConnectionResult => ({
  isValid,
  messageTKey,
});

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

export const buildAcceptResult = (): ValidateConnectionResult => ({ isValid: true });
export const buildRejectResult = (messageTKey: string): ValidateConnectionResult => ({ isValid: false, messageTKey });

export const validateConnection: ValidateConnectionFunc = (c, nodes, edges, templates, ignoreEdge) => {
  if (c.source === c.target) {
    return buildRejectResult('nodes.cannotConnectToSelf');
  }

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
    // Collect nodes shouldn't mix and match field types
    const collectItemType = getCollectItemType(templates, nodes, edges, targetNode.id);
    if (collectItemType) {
      if (!areTypesEqual(sourceFieldTemplate.type, collectItemType)) {
        return buildRejectResult('nodes.cannotMixAndMatchCollectionItemTypes');
      }
    }
  }

  if (
    edges.find((e) => {
      return e.target === c.target && e.targetHandle === c.targetHandle;
    }) &&
    // except CollectionItem inputs can have multiples
    targetFieldTemplate.type.name !== 'CollectionItemField'
  ) {
    return buildRejectResult('nodes.inputMayOnlyHaveOneConnection');
  }

  return buildAcceptResult();
};
