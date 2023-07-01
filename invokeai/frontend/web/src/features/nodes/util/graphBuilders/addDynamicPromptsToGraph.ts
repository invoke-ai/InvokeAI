import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import {
  DynamicPromptInvocation,
  IterateInvocation,
  NoiseInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
} from 'services/api/types';
import {
  DYNAMIC_PROMPT,
  ITERATE,
  NOISE,
  POSITIVE_CONDITIONING,
  RANDOM_INT,
  RANGE_OF_SIZE,
} from './constants';
import { unset } from 'lodash-es';

export const addDynamicPromptsToGraph = (
  graph: NonNullableGraph,
  state: RootState
): void => {
  const { positivePrompt, iterations, seed, shouldRandomizeSeed } =
    state.generation;

  const {
    combinatorial,
    isEnabled: isDynamicPromptsEnabled,
    maxPrompts,
  } = state.dynamicPrompts;

  if (isDynamicPromptsEnabled) {
    // iteration is handled via dynamic prompts
    unset(graph.nodes[POSITIVE_CONDITIONING], 'prompt');

    const dynamicPromptNode: DynamicPromptInvocation = {
      id: DYNAMIC_PROMPT,
      type: 'dynamic_prompt',
      max_prompts: combinatorial ? maxPrompts : iterations,
      combinatorial,
      prompt: positivePrompt,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
    };

    graph.nodes[DYNAMIC_PROMPT] = dynamicPromptNode;
    graph.nodes[ITERATE] = iterateNode;

    // connect dynamic prompts to compel nodes
    graph.edges.push(
      {
        source: {
          node_id: DYNAMIC_PROMPT,
          field: 'prompt_collection',
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

    if (shouldRandomizeSeed) {
      // Random int node to generate the starting seed
      const randomIntNode: RandomIntInvocation = {
        id: RANDOM_INT,
        type: 'rand_int',
      };

      graph.nodes[RANDOM_INT] = randomIntNode;

      // Connect random int to the start of the range of size so the range starts on the random first seed
      graph.edges.push({
        source: { node_id: RANDOM_INT, field: 'a' },
        destination: { node_id: NOISE, field: 'seed' },
      });
    } else {
      // User specified seed, so set the start of the range of size to the seed
      (graph.nodes[NOISE] as NoiseInvocation).seed = seed;
    }
  } else {
    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE,
      type: 'range_of_size',
      size: iterations,
      step: 1,
    };

    const iterateNode: IterateInvocation = {
      id: ITERATE,
      type: 'iterate',
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

    // handle seed
    if (shouldRandomizeSeed) {
      // Random int node to generate the starting seed
      const randomIntNode: RandomIntInvocation = {
        id: RANDOM_INT,
        type: 'rand_int',
      };

      graph.nodes[RANDOM_INT] = randomIntNode;

      // Connect random int to the start of the range of size so the range starts on the random first seed
      graph.edges.push({
        source: { node_id: RANDOM_INT, field: 'a' },
        destination: { node_id: RANGE_OF_SIZE, field: 'start' },
      });
    } else {
      // User specified seed, so set the start of the range of size to the seed
      rangeOfSizeNode.start = seed;
    }
  }
};
