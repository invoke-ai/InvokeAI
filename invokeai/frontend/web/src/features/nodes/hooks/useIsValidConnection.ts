// TODO: enable this at some point
import graphlib from '@dagrejs/graphlib';
import { useAppSelector } from 'app/store/storeHooks';
import { useCallback } from 'react';
import { Connection, Edge, Node, useReactFlow } from 'reactflow';
import { COLLECTION_TYPES } from '../types/constants';
import { InvocationNodeData } from '../types/types';

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

      // Connection types must be the same for a connection
      if (
        sourceType !== targetType &&
        sourceType !== 'CollectionItem' &&
        targetType !== 'CollectionItem'
      ) {
        if (
          !(
            COLLECTION_TYPES.includes(targetType) &&
            COLLECTION_TYPES.includes(sourceType)
          )
        ) {
          return false;
        }
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
