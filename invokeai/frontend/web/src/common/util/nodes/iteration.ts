import { v4 as uuidv4 } from 'uuid';
import {
  Edge,
  Graph,
  ImageToImageInvocation,
  IterateInvocation,
  RandomRangeInvocation,
  RangeInvocation,
  TextToImageInvocation,
} from 'services/api';
import { buildImg2ImgNode } from './image2Image';
import { RootState } from 'app/store';

type BuildIteration = {
  graph: Graph;
  iterations: number;
  shouldRandomizeSeed: boolean;
};

type BuildRangeNodeArg = {
  state: RootState;
};

const buildRangeNode = (
  state: RootState
): Record<string, RangeInvocation | RandomRangeInvocation> => {
  const nodeId = uuidv4();
  const { shouldRandomizeSeed, iterations, seed } = state.generation;

  if (shouldRandomizeSeed) {
    return {
      [nodeId]: {
        id: nodeId,
        type: 'random_range',
        size: iterations,
      },
    };
  }

  return {
    [nodeId]: {
      id: nodeId,
      type: 'range',
      start: seed,
      stop: seed + iterations,
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

export const buildIteration = (graph: Graph, state: RootState): Graph => {
  const rangeNode = buildRangeNode(state);
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
