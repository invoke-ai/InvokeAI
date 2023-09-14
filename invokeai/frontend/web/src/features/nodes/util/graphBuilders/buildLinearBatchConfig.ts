import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { unset } from 'lodash-es';
import { BatchConfig, RandomIntInvocation } from 'services/api/types';
import {
  METADATA_ACCUMULATOR,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
} from './constants';

export const prepareLinearUIBatch = (
  state: RootState,
  graph: NonNullableGraph,
  prepend: boolean
): BatchConfig => {
  const { shouldRandomizeSeed, iterations } = state.generation;

  const { prompts } = state.dynamicPrompts;

  if (shouldRandomizeSeed) {
    const randomIntNode: RandomIntInvocation = {
      id: RANDOM_INT,
      type: 'rand_int',
      is_intermediate: true,
    };
    graph.nodes[RANDOM_INT] = randomIntNode;
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'value' },
      destination: { node_id: NOISE, field: 'seed' },
    });
    graph.edges.push({
      source: { node_id: RANDOM_INT, field: 'value' },
      destination: { node_id: METADATA_ACCUMULATOR, field: 'seed' },
    });

    unset(graph.nodes[NOISE], 'seed');
  }

  const data: BatchConfig['batch']['data'] = [];

  if (prompts.length > 1) {
    unset(graph.nodes[METADATA_ACCUMULATOR], 'positive_prompt');

    // zipped batch of prompts
    data.push([
      {
        node_path: POSITIVE_CONDITIONING,
        field_name: 'prompt',
        items: prompts,
      },
      {
        node_path: METADATA_ACCUMULATOR,
        field_name: 'positive_prompt',
        items: prompts,
      },
    ]);
  }

  const enqueueBatchArg: BatchConfig = {
    prepend,
    batch: {
      graph,
      runs: iterations,
      data: data.length ? data : undefined,
    },
  };

  return enqueueBatchArg;
};
