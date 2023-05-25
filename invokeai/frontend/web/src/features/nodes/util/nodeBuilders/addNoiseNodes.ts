import { RootState } from 'app/store/store';
import {
  IterateInvocation,
  NoiseInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { cloneDeep } from 'lodash-es';

const NOISE = 'noise';
const RANDOM_INT = 'rand_int';
const RANGE_OF_SIZE = 'range_of_size';
const ITERATE = 'iterate';
/**
 * Adds the appropriate noise nodes to a linear UI t2l or l2l graph.
 *
 * @param graph The graph to add the noise nodes to.
 * @param baseNodeId The id of the base node to connect the noise nodes to.
 * @param state The app state..
 */
export const addNoiseNodes = (
  graph: NonNullableGraph,
  baseNodeId: string,
  state: RootState
): NonNullableGraph => {
  const graphClone = cloneDeep(graph);

  // Create and add the noise nodes
  const { width, height, seed, iterations, shouldRandomizeSeed } =
    state.generation;

  // Single iteration, explicit seed
  if (!shouldRandomizeSeed && iterations === 1) {
    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      seed: seed,
      width,
      height,
    };

    graphClone.nodes[NOISE] = noiseNode;

    // Connect them
    graphClone.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: baseNodeId,
        field: 'noise',
      },
    });
  }

  // Single iteration, random seed
  if (shouldRandomizeSeed && iterations === 1) {
    // TODO: This assumes the `high` value is the max seed value
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      width,
      height,
    };

    graphClone.nodes[RANDOM_INT] = randomIntNode;
    graphClone.nodes[NOISE] = noiseNode;

    graphClone.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    graphClone.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: baseNodeId,
        field: 'noise',
      },
    });
  }

  // Multiple iterations, explicit seed
  if (!shouldRandomizeSeed && iterations > 1) {
    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      start: seed,
      size: iterations,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
    };

    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      width,
      height,
    };

    graphClone.nodes[RANGE_OF_SIZE] = rangeOfSizeNode;
    graphClone.nodes[ITERATE] = iterateNode;
    graphClone.nodes[NOISE] = noiseNode;

    graphClone.edges.push({
      source: { node_id: RANGE_OF_SIZE, field: 'collection' },
      destination: {
        node_id: ITERATE,
        field: 'collection',
      },
    });

    graphClone.edges.push({
      source: {
        node_id: ITERATE,
        field: 'item',
      },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    graphClone.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: baseNodeId,
        field: 'noise',
      },
    });
  }

  // Multiple iterations, random seed
  if (shouldRandomizeSeed && iterations > 1) {
    // TODO: This assumes the `high` value is the max seed value
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
    };

    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      size: iterations,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
    };

    const noiseNode: NoiseInvocation = {
      id: NOISE,
      type: 'noise',
      width,
      height,
    };

    graphClone.nodes[RANDOM_INT] = randomIntNode;
    graphClone.nodes[RANGE_OF_SIZE] = rangeOfSizeNode;
    graphClone.nodes[ITERATE] = iterateNode;
    graphClone.nodes[NOISE] = noiseNode;

    graphClone.edges.push({
      source: { node_id: RANDOM_INT, field: 'a' },
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });

    graphClone.edges.push({
      source: { node_id: RANGE_OF_SIZE, field: 'collection' },
      destination: {
        node_id: ITERATE,
        field: 'collection',
      },
    });

    graphClone.edges.push({
      source: {
        node_id: ITERATE,
        field: 'item',
      },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    graphClone.edges.push({
      source: { node_id: NOISE, field: 'noise' },
      destination: {
        node_id: baseNodeId,
        field: 'noise',
      },
    });
  }

  return graphClone;
};
