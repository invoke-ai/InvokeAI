import graphlib from '@dagrejs/graphlib';
import type { Edge, Node } from 'reactflow';

export const getIsGraphAcyclic = (source: string, target: string, nodes: Node[], edges: Edge[]) => {
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
