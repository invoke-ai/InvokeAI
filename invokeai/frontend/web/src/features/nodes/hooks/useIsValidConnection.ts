import { useCallback } from 'react';
import { Connection, Node, useReactFlow } from 'reactflow';
import graphlib from '@dagrejs/graphlib';
import { InvocationValue } from '../types/types';

export const useIsValidConnection = () => {
  const flow = useReactFlow();

  // Check if an in-progress connection is valid
  const isValidConnection = useCallback(
    ({ source, sourceHandle, target, targetHandle }: Connection): boolean => {
      const edges = flow.getEdges();
      const nodes = flow.getNodes();

      return true;

      // // Connection must have valid targets
      // if (!(source && sourceHandle && target && targetHandle)) {
      //   return false;
      // }

      // // Connection is invalid if target already has a connection
      // if (
      //   edges.find((edge) => {
      //     return edge.target === target && edge.targetHandle === targetHandle;
      //   })
      // ) {
      //   return false;
      // }

      // // Find the source and target nodes
      // const sourceNode = flow.getNode(source) as Node<InvocationValue>;

      // const targetNode = flow.getNode(target) as Node<InvocationValue>;

      // // Conditional guards against undefined nodes/handles
      // if (!(sourceNode && targetNode && sourceNode.data && targetNode.data)) {
      //   return false;
      // }

      // // Connection types must be the same for a connection
      // if (
      //   sourceNode.data.outputs[sourceHandle].type !==
      //   targetNode.data.inputs[targetHandle].type
      // ) {
      //   return false;
      // }

      // // Graphs much be acyclic (no loops!)

      // /**
      //  * TODO: use `graphlib.alg.findCycles()` to identify strong connections
      //  *
      //  * this validation func only runs when the cursor hits the second handle of the connection,
      //  * and only on that second handle - so it cannot tell us exhaustively which connections
      //  * are valid.
      //  *
      //  * ideally, we check when the connection starts to calculate all invalid handles at once.
      //  *
      //  * requires making a new graphlib graph - and calling `findCycles()` - for each potential
      //  * handle. instead of using the `isValidConnection` prop, it would use the `onConnectStart`
      //  * prop.
      //  *
      //  * the strong connections should be stored in global state.
      //  *
      //  * then, `isValidConnection` would simple loop through the strong connections and if the
      //  * source and target are in a single strong connection, return false.
      //  *
      //  * and also, we can use this knowledge to style every handle when a connection starts,
      //  * which is otherwise not possible.
      //  */

      // // build a graphlib graph
      // const g = new graphlib.Graph();

      // nodes.forEach((n) => {
      //   g.setNode(n.id);
      // });

      // edges.forEach((e) => {
      //   g.setEdge(e.source, e.target);
      // });

      // // Add the candidate edge to the graph
      // g.setEdge(source, target);

      // return graphlib.alg.isAcyclic(g);
    },
    [flow]
  );

  return isValidConnection;
};
