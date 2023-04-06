import { v4 as uuidv4 } from 'uuid';
import {
  Edge,
  Graph,
  ImageToImageInvocation,
  IterateInvocation,
  RangeInvocation,
  TextToImageInvocation,
} from 'services/api';
import { buildImg2ImgNode } from './image2Image';

type BuildIteration = {
  graph: Graph;
  iterations: number;
};

const buildRangeNode = (
  iterations: number
): Record<string, RangeInvocation> => {
  const nodeId = uuidv4();
  return {
    [nodeId]: {
      id: nodeId,
      type: 'range',
      start: 0,
      stop: iterations,
      step: 1,
    },
  };
};

const buildIterateNode = (): Record<string, IterateInvocation> => {
  const nodeId = uuidv4();
  return {
    [nodeId]: {
      id: nodeId,
      type: 'iterate',
      collection: [],
      index: 0,
    },
  };
};

export const buildIteration = ({
  graph,
  iterations,
}: BuildIteration): Graph => {
  const rangeNode = buildRangeNode(iterations);
  const iterateNode = buildIterateNode();
  const baseNode: Graph['nodes'] = graph.nodes;
  const edges: Edge[] = [
    {
      source: {
        field: 'collection',
        node_id: Object.keys(rangeNode)[0],
      },
      destination: {
        field: 'collection',
        node_id: Object.keys(iterateNode)[0],
      },
    },
    {
      source: {
        field: 'item',
        node_id: Object.keys(iterateNode)[0],
      },
      destination: {
        field: 'seed',
        node_id: Object.keys(baseNode!)[0],
      },
    },
  ];
  return {
    nodes: {
      ...rangeNode,
      ...iterateNode,
      ...graph.nodes,
    },
    edges,
  };
};
