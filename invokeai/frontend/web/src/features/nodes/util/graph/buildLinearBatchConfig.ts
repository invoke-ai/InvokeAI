import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { range } from 'es-toolkit/compat';
import type { SeedBehaviour } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { API_BASE_MODELS } from 'features/parameters/types/constants';
import type { components } from 'services/api/schema';
import type { Batch, EnqueueBatchArg, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const getExtendedPrompts = (arg: {
  seedBehaviour: SeedBehaviour;
  iterations: number;
  prompts: string[];
  model: ModelIdentifierField;
}): string[] => {
  const { seedBehaviour, iterations, prompts, model } = arg;
  // Normally, the seed behaviour implicity determines the batch size. But when we use models without seeds (like
  // ChatGPT 4o) in conjunction with the per-prompt seed behaviour, we lose out on that implicit batch size. To rectify
  // this, we need to create a batch of the right size by repeating the prompts.
  if (seedBehaviour === 'PER_PROMPT' || API_BASE_MODELS.includes(model.base)) {
    return range(iterations).flatMap(() => prompts);
  }
  return prompts;
};

export const prepareLinearUIBatch = (arg: {
  state: RootState;
  g: Graph;
  prepend: boolean;
  positivePromptNode: Invocation<'string'>;
  seedNode?: Invocation<'integer'>;
  origin: string;
  destination: string;
}): EnqueueBatchArg => {
  const { state, g, prepend, positivePromptNode, seedNode, origin, destination } = arg;
  const { iterations, model, shouldRandomizeSeed, seed } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  assert(model, 'No model found in state when preparing batch');

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

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

  const extendedPrompts = getExtendedPrompts({ seedBehaviour, iterations, prompts, model });

  // zipped batch of prompts
  firstBatchDatumList.push({
    node_path: positivePromptNode.id,
    field_name: 'value',
    items: extendedPrompts,
  });

  data.push(firstBatchDatumList);

  const enqueueBatchArg: EnqueueBatchArg = {
    prepend,
    batch: {
      graph: g.getGraph(),
      runs: 1,
      data,
      origin,
      destination,
    },
  };

  return enqueueBatchArg;
};
