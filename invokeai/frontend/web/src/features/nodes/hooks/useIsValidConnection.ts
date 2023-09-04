// TODO: enable this at some point
import graphlib from '@dagrejs/graphlib';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { Connection, Edge, Node, useReactFlow } from 'reactflow';
import {
  COLLECTION_MAP,
  COLLECTION_TYPES,
  POLYMORPHIC_TO_SINGLE_MAP,
  POLYMORPHIC_TYPES,
} from '../types/constants';
import { InvocationNodeData } from '../types/types';

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/store/util/makeIsConnectionValidSelector.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const useIsValidConnection = () => {
  const flow = useReactFlow();
  const shouldValidateGraph = useAppSelector(
    (state) => state.nodes.shouldValidateGraph
  );
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      if (!shouldValidateGraph) {
        // manual override!
        return true;
      }

      const edges = flow.getEdges();
      const nodes = flow.getNodes();
      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }

      // Find the source and target nodes
      const sourceNode = flow.getNode(source) as Node<InvocationNodeData>;
      const targetNode = flow.getNode(target) as Node<InvocationNodeData>;

      // Conditional guards against undefined nodes/handles
      if (!(sourceNode && targetNode && sourceNode.data && targetNode.data)) {
        return false;
      }

      const sourceType = sourceNode.data.outputs[sourceHandle]?.type;
      const targetType = targetNode.data.inputs[targetHandle]?.type;

      if (!sourceType || !targetType) {
        // something has gone terribly awry
        return false;
      }

      if (
        edges
          .filter((edge) => {
            return edge.target === target && edge.targetHandle === targetHandle;
          })
          .find((edge) => {
            edge.source === source && edge.sourceHandle === sourceHandle;
          })
      ) {
        // We already have a connection from this source to this target
        return false;
      }

      // Connection is invalid if target already has a connection
      if (
        edges.find((edge) => {
          return edge.target === target && edge.targetHandle === targetHandle;
        }) &&
        // except CollectionItem inputs can have multiples
        targetType !== 'CollectionItem'
      ) {
        return false;
      }

      /**
       * Connection types must be the same for a connection, with exceptions:
       * - CollectionItem can connect to any non-Collection
       * - Non-Collections can connect to CollectionItem
       * - Anything (non-Collections, Collections, Polymorphics) can connect to Polymorphics of the same base type
       * - Generic Collection can connect to any other Collection or Polymorphic
       * - Any Collection can connect to a Generic Collection
       */

      if (sourceType !== targetType) {
        const isCollectionItemToNonCollection =
          sourceType === 'CollectionItem' &&
          !COLLECTION_TYPES.includes(targetType);

        const isNonCollectionToCollectionItem =
          targetType === 'CollectionItem' &&
          !COLLECTION_TYPES.includes(sourceType) &&
          !POLYMORPHIC_TYPES.includes(sourceType);

        const isAnythingToPolymorphicOfSameBaseType =
          POLYMORPHIC_TYPES.includes(targetType) &&
          (() => {
            if (!POLYMORPHIC_TYPES.includes(targetType)) {
              return false;
            }
            const baseType =
              POLYMORPHIC_TO_SINGLE_MAP[
                targetType as keyof typeof POLYMORPHIC_TO_SINGLE_MAP
              ];

            const collectionType =
              COLLECTION_MAP[baseType as keyof typeof COLLECTION_MAP];

            return sourceType === baseType || sourceType === collectionType;
          })();

        const isGenericCollectionToAnyCollectionOrPolymorphic =
          sourceType === 'Collection' &&
          (COLLECTION_TYPES.includes(targetType) ||
            POLYMORPHIC_TYPES.includes(targetType));

        const isCollectionToGenericCollection =
          targetType === 'Collection' && COLLECTION_TYPES.includes(sourceType);

        const isIntToFloat = sourceType === 'integer' && targetType === 'float';

        return (
          isCollectionItemToNonCollection ||
          isNonCollectionToCollectionItem ||
          isAnythingToPolymorphicOfSameBaseType ||
          isGenericCollectionToAnyCollectionOrPolymorphic ||
          isCollectionToGenericCollection ||
          isIntToFloat
        );
      }

      // Graphs much be acyclic (no loops!)
      return getIsGraphAcyclic(source, target, nodes, edges);
    },
    [flow, shouldValidateGraph]
  );

  return isValidConnection;
};

export const getIsGraphAcyclic = (
  source: string,
  target: string,
  nodes: Node[],
  edges: Edge[]
) => {
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
  return graphlib.alg.isAcyclic(g);
};
