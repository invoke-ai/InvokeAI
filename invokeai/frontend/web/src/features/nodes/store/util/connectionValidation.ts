import graphlib from '@dagrejs/graphlib';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import type { RootState } from 'app/store/store';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import type { NodesState, PendingConnection, Templates } from 'features/nodes/store/types';
import { type FieldType, isStatefulFieldType } from 'features/nodes/types/field';
import type { AnyNode, InvocationNode, InvocationNodeEdge, InvocationTemplate } from 'features/nodes/types/invocation';
import i18n from 'i18next';
import { differenceWith, isEqual, map, omit } from 'lodash-es';
import type { Connection, Edge, HandleType, Node } from 'reactflow';
import { assert } from 'tsafe';

/**
 * Finds the first valid field for a pending connection between two nodes.
 * @param templates The invocation templates
 * @param nodes The current nodes
 * @param edges The current edges
 * @param pendingConnection The pending connection
 * @param candidateNode The candidate node to which the connection is being made
 * @param candidateTemplate The candidate template for the candidate node
 * @returns The first valid connection, or null if no valid connection is found
 */
export const getFirstValidConnection = (
  templates: Templates,
  nodes: AnyNode[],
  edges: InvocationNodeEdge[],
  pendingConnection: PendingConnection,
  candidateNode: InvocationNode,
  candidateTemplate: InvocationTemplate
): Connection | null => {
  if (pendingConnection.node.id === candidateNode.id) {
    // Cannot connect to self
    return null;
  }

  const pendingFieldKind = pendingConnection.fieldTemplate.fieldKind === 'input' ? 'target' : 'source';

  if (pendingFieldKind === 'source') {
    // Connecting from a source to a target
    if (getHasCycles(pendingConnection.node.id, candidateNode.id, nodes, edges)) {
      return null;
    }
    if (candidateNode.data.type === 'collect') {
      // Special handling for collect node - the `item` field takes any number of connections
      return {
        source: pendingConnection.node.id,
        sourceHandle: pendingConnection.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: 'item',
      };
    }
    // Only one connection per target field is allowed - look for an unconnected target field
    const candidateFields = map(candidateTemplate.inputs);
    const candidateConnectedFields = edges
      .filter((edge) => edge.target === candidateNode.id)
      .map((edge) => {
        // Edges must always have a targetHandle, safe to assert here
        assert(edge.targetHandle);
        return edge.targetHandle;
      });
    const candidateUnconnectedFields = differenceWith(
      candidateFields,
      candidateConnectedFields,
      (field, connectedFieldName) => field.name === connectedFieldName
    );
    const candidateField = candidateUnconnectedFields.find((field) =>
      validateSourceAndTargetTypes(pendingConnection.fieldTemplate.type, field.type)
    );
    if (candidateField) {
      return {
        source: pendingConnection.node.id,
        sourceHandle: pendingConnection.fieldTemplate.name,
        target: candidateNode.id,
        targetHandle: candidateField.name,
      };
    }
  } else {
    // Connecting from a target to a source
    // Ensure we there is not already an edge to the target, except for collect nodes
    const isCollect = pendingConnection.node.data.type === 'collect';
    const isTargetAlreadyConnected = edges.some(
      (e) => e.target === pendingConnection.node.id && e.targetHandle === pendingConnection.fieldTemplate.name
    );
    if (!isCollect && isTargetAlreadyConnected) {
      return null;
    }

    if (getHasCycles(candidateNode.id, pendingConnection.node.id, nodes, edges)) {
      return null;
    }

    // Sources/outputs can have any number of edges, we can take the first matching output field
    let candidateFields = map(candidateTemplate.outputs);
    if (isCollect) {
      // Narrow candidates to same field type as already is connected to the collect node
      const collectItemType = getCollectItemType(templates, nodes, edges, pendingConnection.node.id);
      if (collectItemType) {
        candidateFields = candidateFields.filter((field) => areTypesEqual(field.type, collectItemType));
      }
    }
    const candidateField = candidateFields.find((field) => {
      const isValid = validateSourceAndTargetTypes(field.type, pendingConnection.fieldTemplate.type);
      const isAlreadyConnected = edges.some((e) => e.source === candidateNode.id && e.sourceHandle === field.name);
      return isValid && !isAlreadyConnected;
    });
    if (candidateField) {
      return {
        source: candidateNode.id,
        sourceHandle: candidateField.name,
        target: pendingConnection.node.id,
        targetHandle: pendingConnection.fieldTemplate.name,
      };
    }
  }

  return null;
};

/**
 * Check if adding an edge between the source and target nodes would create a cycle in the graph.
 * @param source The source node id
 * @param target The target node id
 * @param nodes The graph's current nodes
 * @param edges The graph's current edges
 * @returns True if the graph would be acyclic after adding the edge, false otherwise
 */
export const getHasCycles = (source: string, target: string, nodes: Node[], edges: Edge[]) => {
  // construct graphlib graph from editor state
  const g = new graphlib.Graph();

  nodes.forEach((n) => {
    g.setNode(n.id);
  });

  edges.forEach((e) => {
    g.setEdge(e.source, e.target);
  });

  // add the candidate edge
  g.setEdge(source, target);

  // check if the graph is acyclic
  return !graphlib.alg.isAcyclic(g);
};

/**
 * Given a collect node, return the type of the items it collects. The graph is traversed to find the first node and
 * field connected to the collector's `item` input. The field type of that field is returned, else null if there is no
 * input field.
 * @param templates The current invocation templates
 * @param nodes The current nodes
 * @param edges The current edges
 * @param nodeId The collect node's id
 * @returns The type of the items the collect node collects, or null if there is no input field
 */
export const getCollectItemType = (
  templates: Templates,
  nodes: AnyNode[],
  edges: InvocationNodeEdge[],
  nodeId: string
): FieldType | null => {
  const firstEdgeToCollect = edges.find((edge) => edge.target === nodeId && edge.targetHandle === 'item');
  if (!firstEdgeToCollect?.sourceHandle) {
    return null;
  }
  const node = nodes.find((n) => n.id === firstEdgeToCollect.source);
  if (!node) {
    return null;
  }
  const template = templates[node.data.type];
  if (!template) {
    return null;
  }
  const fieldType = template.outputs[firstEdgeToCollect.sourceHandle]?.type ?? null;
  return fieldType;
};

/**
 * Creates a selector that validates a pending connection.
 *
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/hooks/useIsValidConnection.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 *
 * @param templates The invocation templates
 * @param pendingConnection The current pending connection (if there is one)
 * @param nodeId The id of the node for which the selector is being created
 * @param fieldName The name of the field for which the selector is being created
 * @param handleType The type of the handle for which the selector is being created
 * @param fieldType The type of the field for which the selector is being created
 * @returns
 */
export const makeConnectionErrorSelector = (
  templates: Templates,
  nodeId: string,
  fieldName: string,
  handleType: HandleType,
  fieldType: FieldType
) => {
  return createMemoizedSelector(
    selectNodesSlice,
    (state: RootState, pendingConnection: PendingConnection | null) => pendingConnection,
    (nodesSlice: NodesState, pendingConnection: PendingConnection | null) => {
      const { nodes, edges } = nodesSlice;

      if (!pendingConnection) {
        return i18n.t('nodes.noConnectionInProgress');
      }

      const connectionNodeId = pendingConnection.node.id;
      const connectionFieldName = pendingConnection.fieldTemplate.name;
      const connectionHandleType = pendingConnection.fieldTemplate.fieldKind === 'input' ? 'target' : 'source';
      const connectionStartFieldType = pendingConnection.fieldTemplate.type;

      if (!connectionHandleType || !connectionNodeId || !connectionFieldName) {
        return i18n.t('nodes.noConnectionData');
      }

      const targetType = handleType === 'target' ? fieldType : connectionStartFieldType;
      const sourceType = handleType === 'source' ? fieldType : connectionStartFieldType;

      if (nodeId === connectionNodeId) {
        return i18n.t('nodes.cannotConnectToSelf');
      }

      if (handleType === connectionHandleType) {
        if (handleType === 'source') {
          return i18n.t('nodes.cannotConnectOutputToOutput');
        }
        return i18n.t('nodes.cannotConnectInputToInput');
      }

      // we have to figure out which is the target and which is the source
      const targetNodeId = handleType === 'target' ? nodeId : connectionNodeId;
      const targetFieldName = handleType === 'target' ? fieldName : connectionFieldName;
      const sourceNodeId = handleType === 'source' ? nodeId : connectionNodeId;
      const sourceFieldName = handleType === 'source' ? fieldName : connectionFieldName;

      if (
        edges.find((edge) => {
          edge.target === targetNodeId &&
            edge.targetHandle === targetFieldName &&
            edge.source === sourceNodeId &&
            edge.sourceHandle === sourceFieldName;
        })
      ) {
        // We already have a connection from this source to this target
        return i18n.t('nodes.cannotDuplicateConnection');
      }

      const targetNode = nodes.find((node) => node.id === targetNodeId);
      assert(targetNode, `Target node not found: ${targetNodeId}`);
      const targetTemplate = templates[targetNode.data.type];
      assert(targetTemplate, `Target template not found: ${targetNode.data.type}`);

      if (targetTemplate.inputs[targetFieldName]?.input === 'direct') {
        return i18n.t('nodes.cannotConnectToDirectInput');
      }

      if (targetNode.data.type === 'collect' && targetFieldName === 'item') {
        // Collect nodes shouldn't mix and match field types
        const collectItemType = getCollectItemType(templates, nodes, edges, targetNode.id);
        if (collectItemType) {
          if (!areTypesEqual(sourceType, collectItemType)) {
            return i18n.t('nodes.cannotMixAndMatchCollectionItemTypes');
          }
        }
      }

      if (
        edges.find((edge) => {
          return edge.target === targetNodeId && edge.targetHandle === targetFieldName;
        }) &&
        // except CollectionItem inputs can have multiples
        targetType.name !== 'CollectionItemField'
      ) {
        return i18n.t('nodes.inputMayOnlyHaveOneConnection');
      }

      if (!validateSourceAndTargetTypes(sourceType, targetType)) {
        return i18n.t('nodes.fieldTypesMustMatch');
      }

      const hasCycles = getHasCycles(
        connectionHandleType === 'source' ? connectionNodeId : nodeId,
        connectionHandleType === 'source' ? nodeId : connectionNodeId,
        nodes,
        edges
      );

      if (hasCycles) {
        return i18n.t('nodes.connectionWouldCreateCycle');
      }

      return;
    }
  );
};

/**
 * Validates that the source and target types are compatible for a connection.
 * @param sourceType The type of the source field.
 * @param targetType The type of the target field.
 * @returns True if the connection is valid, false otherwise.
 */
export const validateSourceAndTargetTypes = (sourceType: FieldType, targetType: FieldType) => {
  // TODO: There's a bug with Collect -> Iterate nodes:
  // https://github.com/invoke-ai/InvokeAI/issues/3956
  // Once this is resolved, we can remove this check.
  if (sourceType.name === 'CollectionField' && targetType.name === 'CollectionField') {
    return false;
  }

  if (areTypesEqual(sourceType, targetType)) {
    return true;
  }

  /**
   * Connection types must be the same for a connection, with exceptions:
   * - CollectionItem can connect to any non-Collection
   * - Non-Collections can connect to CollectionItem
   * - Anything (non-Collections, Collections, CollectionOrScalar) can connect to CollectionOrScalar of the same base type
   * - Generic Collection can connect to any other Collection or CollectionOrScalar
   * - Any Collection can connect to a Generic Collection
   */
  const isCollectionItemToNonCollection = sourceType.name === 'CollectionItemField' && !targetType.isCollection;

  const isNonCollectionToCollectionItem =
    targetType.name === 'CollectionItemField' && !sourceType.isCollection && !sourceType.isCollectionOrScalar;

  const isAnythingToCollectionOrScalarOfSameBaseType =
    targetType.isCollectionOrScalar && sourceType.name === targetType.name;

  const isGenericCollectionToAnyCollectionOrCollectionOrScalar =
    sourceType.name === 'CollectionField' && (targetType.isCollection || targetType.isCollectionOrScalar);

  const isCollectionToGenericCollection = targetType.name === 'CollectionField' && sourceType.isCollection;

  const areBothTypesSingle =
    !sourceType.isCollection &&
    !sourceType.isCollectionOrScalar &&
    !targetType.isCollection &&
    !targetType.isCollectionOrScalar;

  const isIntToFloat = areBothTypesSingle && sourceType.name === 'IntegerField' && targetType.name === 'FloatField';

  const isIntOrFloatToString =
    areBothTypesSingle &&
    (sourceType.name === 'IntegerField' || sourceType.name === 'FloatField') &&
    targetType.name === 'StringField';

  const isTargetAnyType = targetType.name === 'AnyField';

  // One of these must be true for the connection to be valid
  return (
    isCollectionItemToNonCollection ||
    isNonCollectionToCollectionItem ||
    isAnythingToCollectionOrScalarOfSameBaseType ||
    isGenericCollectionToAnyCollectionOrCollectionOrScalar ||
    isCollectionToGenericCollection ||
    isIntToFloat ||
    isIntOrFloatToString ||
    isTargetAnyType
  );
};

/**
 * Checks if two types are equal. If the field types have original types, those are also compared. Any match is
 * considered equal. For example, if the source type and original target type match, the types are considered equal.
 * @param sourceType The type of the source field.
 * @param targetType The type of the target field.
 * @returns True if the types are equal, false otherwise.
 */
export const areTypesEqual = (sourceType: FieldType, targetType: FieldType) => {
  const _sourceType = isStatefulFieldType(sourceType) ? omit(sourceType, 'originalType') : sourceType;
  const _targetType = isStatefulFieldType(targetType) ? omit(targetType, 'originalType') : targetType;
  const _sourceTypeOriginal = isStatefulFieldType(sourceType) ? sourceType.originalType : sourceType;
  const _targetTypeOriginal = isStatefulFieldType(targetType) ? targetType.originalType : targetType;
  if (isEqual(_sourceType, _targetType)) {
    return true;
  }
  if (_targetTypeOriginal && isEqual(_sourceType, _targetTypeOriginal)) {
    return true;
  }
  if (_sourceTypeOriginal && isEqual(_sourceTypeOriginal, _targetType)) {
    return true;
  }
  if (_sourceTypeOriginal && _targetTypeOriginal && isEqual(_sourceTypeOriginal, _targetTypeOriginal)) {
    return true;
  }
  return false;
};
