import { Graph } from 'services/api';
import { v4 as uuidv4 } from 'uuid';
import { reduce } from 'lodash-es';
import { RootState } from 'app/store/store';
import { AnyInvocation } from 'services/events/types';

/**
 * Builds a graph from the node editor state.
 */
export const buildNodesGraph = (state: RootState): Graph => {
  const { nodes, edges } = state.nodes;

  // Reduce the node editor nodes into invocation graph nodes
  const parsedNodes = nodes.reduce<NonNullable<Graph['nodes']>>(
    (nodesAccumulator, node, nodeIndex) => {
      const { id, data } = node;
      const { type, inputs } = data;

      // Transform each node's inputs to simple key-value pairs
      const transformedInputs = reduce(
        inputs,
        (inputsAccumulator, input, name) => {
          inputsAccumulator[name] = input.value;

          return inputsAccumulator;
        },
        {} as Record<Exclude<string, 'id' | 'type'>, any>
      );

      // Build this specific node
      const graphNode = {
        type,
        id,
        ...transformedInputs,
      };

      // Add it to the nodes object
      Object.assign(nodesAccumulator, {
        [id]: graphNode,
      });

      return nodesAccumulator;
    },
    {}
  );

  // Reduce the node editor edges into invocation graph edges
  const parsedEdges = edges.reduce<NonNullable<Graph['edges']>>(
    (edgesAccumulator, edge, edgeIndex) => {
      const { source, target, sourceHandle, targetHandle } = edge;

      // Format the edges and add to the edges array
      edgesAccumulator.push({
        source: {
          node_id: source,
          field: sourceHandle as string,
        },
        destination: {
          node_id: target,
          field: targetHandle as string,
        },
      });

      return edgesAccumulator;
    },
    []
  );

  // Assemble!
  const graph = {
    id: uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };

  return graph;
};
