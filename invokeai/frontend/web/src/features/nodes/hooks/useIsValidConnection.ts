import { useCallback } from 'react';
import { Connection, useReactFlow } from 'reactflow';
import graphlib from '@dagrejs/graphlib';

export const useIsValidConnection = () => {
  const flow = useReactFlow();

  // Check if an in-progress connection is valid
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      return true;
      const edges = flow.getEdges();
      const nodes = flow.getNodes();

      // Connection must have valid targets
      if (!(source && sourceHandle && target && targetHandle)) {
        return false;
      }

      // Connection is invalid if target already has a connection
      if (
        edges.find((edge) => {
          return edge.target === target && edge.targetHandle === targetHandle;
        })
      ) {
        return false;
      }

      // Find the source and target nodes
      const sourceNode = flow.getNode(source);
      const targetNode = flow.getNode(target);

      // Conditional guards against undefined nodes/handles
      if (!(sourceNode && targetNode)) {
        return false;
      }

      // Connection types must be the same for a connection
      if (
        sourceNode.data.outputs[sourceHandle].type !==
        targetNode.data.inputs[targetHandle].type
      ) {
        return false;
      }

      // Graphs much be acyclic (no loops!)

      // build a graphlib graph
      const g = new graphlib.Graph();

      nodes.forEach((n) => {
        g.setNode(n.id);
      });

      edges.forEach((e) => {
        g.setEdge(e.source, e.target);
      });

      // Add the candidate edge to the graph
      g.setEdge(source, target);

      return graphlib.alg.isAcyclic(g);
    },
    [flow]
  );

  return isValidConnection;
};
