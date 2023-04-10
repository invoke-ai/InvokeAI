import { Edge } from 'reactflow';

export type AdjacencyList = { [nodeId: string]: string[] };

export const buildAdjacencyList = (edges: Edge[]): AdjacencyList => {
  const adjacencyList: AdjacencyList = {};

  edges.forEach((edge) => {
    if (!adjacencyList[edge.source]) {
      adjacencyList[edge.source] = [];
    }

    if (!adjacencyList[edge.target]) {
      adjacencyList[edge.target] = [];
    }

    adjacencyList[edge.source].push(edge.target);
  });

  return adjacencyList;
};
