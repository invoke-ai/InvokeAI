// TODO: enable this at some point
import { useAppSelector, useAppStore } from 'app/store/storeHooks';
import { getIsGraphAcyclic } from 'features/nodes/store/util/getIsGraphAcyclic';
import { validateSourceAndTargetTypes } from 'features/nodes/store/util/validateSourceAndTargetTypes';
import type { InvocationNodeData } from 'features/nodes/types/invocation';
import { useCallback } from 'react';
import type { Connection, Node } from 'reactflow';

/**
 * NOTE: The logic here must be duplicated in `invokeai/frontend/web/src/features/nodes/store/util/makeIsConnectionValidSelector.ts`
 * TODO: Figure out how to do this without duplicating all the logic
 */

export const useIsValidConnection = () => {
  const store = useAppStore();
  const shouldValidateGraph = useAppSelector((s) => s.nodes.shouldValidateGraph);
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }

      if (source === target) {
        // Don't allow nodes to connect to themselves, even if validation is disabled
        return false;
      }

      const state = store.getState();
      const { nodes, edges, templates } = state.nodes;

      // Find the source and target nodes
      const sourceNode = nodes.find((node) => node.id === source) as Node<InvocationNodeData>;
      const targetNode = nodes.find((node) => node.id === target) as Node<InvocationNodeData>;
      const sourceFieldTemplate = templates[sourceNode.data.type]?.outputs[sourceHandle];
      const targetFieldTemplate = templates[targetNode.data.type]?.inputs[targetHandle];

      // Conditional guards against undefined nodes/handles
      if (!(sourceFieldTemplate && targetFieldTemplate)) {
        return false;
      }

      if (!shouldValidateGraph) {
        // manual override!
        return true;
      }

      if (
        edges.find((edge) => {
          edge.target === target &&
            edge.targetHandle === targetHandle &&
            edge.source === source &&
            edge.sourceHandle === sourceHandle;
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
        targetFieldTemplate.type.name !== 'CollectionItemField'
      ) {
        return false;
      }

      // Must use the originalType here if it exists
      if (!validateSourceAndTargetTypes(sourceFieldTemplate.type, targetFieldTemplate.type)) {
        return false;
      }

      // Graphs much be acyclic (no loops!)
      return getIsGraphAcyclic(source, target, nodes, edges);
    },
    [shouldValidateGraph, store]
  );

  return isValidConnection;
};
