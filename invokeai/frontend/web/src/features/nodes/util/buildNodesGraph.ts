import { Graph } from 'services/api';
import { v4 as uuidv4 } from 'uuid';
import { reduce } from 'lodash';
import { RootState } from 'app/store';

export const buildNodesGraph = (state: RootState): Graph => {
  const { nodes, edges } = state.nodes;

  const parsedNodes = nodes.reduce<NonNullable<Graph['nodes']>>(
    (nodesAccumulator, node, nodeIndex) => {
      const { id, data } = node;
      const { type, inputs } = data;

      const transformedInputs = reduce(
        inputs,
        (inputsAccumulator, input, name) => {
          inputsAccumulator[name] = input.value;

          return inputsAccumulator;
        },
        {} as Record<string, any>
      );

      const graphNode = {
        type,
        id,
        ...transformedInputs,
      };

      nodesAccumulator[id] = graphNode;

      return nodesAccumulator;
    },
    {}
  );

  const parsedEdges = edges.reduce<NonNullable<Graph['edges']>>(
    (edgesAccumulator, edge, edgeIndex) => {
      const { source, target, sourceHandle, targetHandle } = edge;

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

  const graph = {
    id: uuidv4(),
    nodes: parsedNodes,
    edges: parsedEdges,
  };

  return graph;
};
