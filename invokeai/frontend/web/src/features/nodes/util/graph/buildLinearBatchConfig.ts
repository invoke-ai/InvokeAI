import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import type { BaseModelType } from 'features/nodes/types/common';
import {
  getExtendedDynamicPrompts,
  getShouldUsePerOutputDynamicPrompts,
} from 'features/nodes/util/graph/dynamicPromptBatching';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { components } from 'services/api/schema';
import type { Batch, EnqueueBatchArg, Invocation } from 'services/api/types';

export const prepareLinearUIBatch = (arg: {
  state: RootState;
  g: Graph;
  prepend: boolean;
  base: BaseModelType;
  positivePromptNode: Invocation<'string'>;
  seedNode?: Invocation<'integer'>;
  origin: string;
  destination: string;
}): EnqueueBatchArg => {
  const { state, g, prepend, positivePromptNode, seedNode, origin, destination } = arg;
  const { iterations, shouldRandomizeSeed, seed } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

  if (getShouldUsePerOutputDynamicPrompts(state)) {
    const perImageBatchDatumList: components['schemas']['BatchDatum'][] = [];

    if (seedNode) {
      // Per-output dynamic prompt mode has already resolved the final queued outputs, so seeds are zipped one-to-one
      // with those outputs instead of applying seedBehaviour as a second batching dimension.
      perImageBatchDatumList.push({
        node_path: seedNode.id,
        field_name: 'value',
        items: generateSeeds({
          count: prompts.length,
          start: shouldRandomizeSeed ? undefined : seed,
        }),
      });
    }

    perImageBatchDatumList.push({
      node_path: positivePromptNode.id,
      field_name: 'value',
      items: prompts,
    });

    data.push(perImageBatchDatumList);

    return {
      prepend,
      batch: {
        graph: g.getGraph(),
        runs: 1,
        data,
        origin,
        destination,
      },
    };
  }

  // add seeds first to ensure the output order groups the prompts
  if (seedNode && seedBehaviour === 'PER_PROMPT') {
    const seeds = generateSeeds({
      count: prompts.length * iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    firstBatchDatumList.push({
      node_path: seedNode.id,
      field_name: 'value',
      items: seeds,
    });
  } else if (seedNode && seedBehaviour === 'PER_ITERATION') {
    // seedBehaviour = SeedBehaviour.PerRun
    const seeds = generateSeeds({
      count: iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    secondBatchDatumList.push({
      node_path: seedNode.id,
      field_name: 'value',
      items: seeds,
    });
    data.push(secondBatchDatumList);
  }

  const extendedPrompts = getExtendedDynamicPrompts({ seedBehaviour, iterations, prompts });

  // zipped batch of prompts
  firstBatchDatumList.push({
    node_path: positivePromptNode.id,
    field_name: 'value',
    items: extendedPrompts,
  });

  data.push(firstBatchDatumList);

  // Models without a seed node (e.g. external API models without seed support) can't express
  // PER_ITERATION via a seed batch dimension, so use batch.runs to repeat the graph instead.
  const runs = !seedNode && seedBehaviour === 'PER_ITERATION' ? iterations : 1;

  const enqueueBatchArg: EnqueueBatchArg = {
    prepend,
    batch: {
      graph: g.getGraph(),
      runs,
      data,
      origin,
      destination,
    },
  };

  return enqueueBatchArg;
};
