import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { unset } from 'lodash-es';
import {
  DynamicPromptInvocation,
  IterateInvocation,
  MetadataAccumulatorInvocation,
  NoiseInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api/types';
import {
  DYNAMIC_PROMPT,
  ITERATE,
  METADATA_ACCUMULATOR,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
} from './constants';

export const addDynamicPromptsToGraph = (
  state: RootState,
  graph: NonNullableGraph
): void => {
  const { positivePrompt, iterations, seed, shouldRandomizeSeed } =
    state.generation;

  const {
    combinatorial,
    isEnabled: isDynamicPromptsEnabled,
    maxPrompts,
  } = state.dynamicPrompts;

  const metadataAccumulator = graph.nodes[METADATA_ACCUMULATOR] as
    | MetadataAccumulatorInvocation
    | undefined;

  if (isDynamicPromptsEnabled) {
    // iteration is handled via dynamic prompts
    unset(graph.nodes[POSITIVE_CONDITIONING], 'prompt');

    const dynamicPromptNode: DynamicPromptInvocation = {
      id: DYNAMIC_PROMPT,
      type: 'dynamic_prompt',
      is_intermediate: true,
      max_prompts: combinatorial ? maxPrompts : iterations,
      combinatorial,
      prompt: positivePrompt,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
      is_intermediate: true,
    };

    graph.nodes[DYNAMIC_PROMPT] = dynamicPromptNode;
    graph.nodes[ITERATE] = iterateNode;

    // connect dynamic prompts to compel nodes
    graph.edges.push(
      {
        source: {
          node_id: DYNAMIC_PROMPT,
          field: 'collection',
        },
        destination: {
          node_id: ITERATE,
          field: 'collection',
        },
      },
      {
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'prompt',
        },
      }
    );

    // hook up positive prompt to metadata
    if (metadataAccumulator) {
      graph.edges.push({
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: METADATA_ACCUMULATOR,
          field: 'positive_prompt',
        },
      });
    }

    if (shouldRandomizeSeed) {
      // Random int node to generate the starting seed
      const randomIntNode: RandomIntInvocation = {
        id: RANDOM_INT,
        type: 'rand_int',
        is_intermediate: true,
      };

      graph.nodes[RANDOM_INT] = randomIntNode;

      // Connect random int to the start of the range of size so the range starts on the random first seed
      graph.edges.push({
        source: { node_id: RANDOM_INT, field: 'value' },
        destination: { node_id: NOISE, field: 'seed' },
      });

      if (metadataAccumulator) {
        graph.edges.push({
          source: { node_id: RANDOM_INT, field: 'value' },
          destination: { node_id: METADATA_ACCUMULATOR, field: 'seed' },
        });
      }
    } else {
      // User specified seed, so set the start of the range of size to the seed
      (graph.nodes[NOISE] as NoiseInvocation).seed = seed;

      // hook up seed to metadata
      if (metadataAccumulator) {
        metadataAccumulator.seed = seed;
      }
    }
  } else {
    // no dynamic prompt - hook up positive prompt
    if (metadataAccumulator) {
      metadataAccumulator.positive_prompt = positivePrompt;
    }

    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      is_intermediate: true,
      size: iterations,
      step: 1,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
      is_intermediate: true,
    };

    graph.nodes[ITERATE] = iterateNode;
    graph.nodes[RANGE_OF_SIZE] = rangeOfSizeNode;

    graph.edges.push({
      source: {
        node_id: RANGE_OF_SIZE,
        field: 'collection',
      },
      destination: {
        node_id: ITERATE,
        field: 'collection',
      },
    });

    graph.edges.push({
      source: {
        node_id: ITERATE,
        field: 'item',
      },
      destination: {
        node_id: NOISE,
        field: 'seed',
      },
    });

    // hook up seed to metadata
    if (metadataAccumulator) {
      graph.edges.push({
        source: {
          node_id: ITERATE,
          field: 'item',
        },
        destination: {
          node_id: METADATA_ACCUMULATOR,
          field: 'seed',
        },
      });
    }
    // handle seed
    if (shouldRandomizeSeed) {
      // Random int node to generate the starting seed
      const randomIntNode: RandomIntInvocation = {
        id: RANDOM_INT,
        type: 'rand_int',
        is_intermediate: true,
      };

      graph.nodes[RANDOM_INT] = randomIntNode;

      // Connect random int to the start of the range of size so the range starts on the random first seed
      graph.edges.push({
        source: { node_id: RANDOM_INT, field: 'value' },
        destination: { node_id: RANGE_OF_SIZE, field: 'start' },
      });
    } else {
      // User specified seed, so set the start of the range of size to the seed
      rangeOfSizeNode.start = seed;
    }
  }
};
