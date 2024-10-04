import graphlib from '@dagrejs/graphlib';
import type { Edge, Node } from 'reactflow';

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
