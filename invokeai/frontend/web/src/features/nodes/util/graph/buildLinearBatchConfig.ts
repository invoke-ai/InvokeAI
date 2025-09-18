import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { range } from 'es-toolkit/compat';
import type { SeedBehaviour } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { API_BASE_MODELS, VIDEO_BASE_MODELS } from 'features/modelManagerV2/models';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { components } from 'services/api/schema';
import type { BaseModelType, Batch, EnqueueBatchArg, Invocation } from 'services/api/types';

const getExtendedPrompts = (arg: {
  seedBehaviour: SeedBehaviour;
  iterations: number;
  prompts: string[];
  base: BaseModelType;
}): string[] => {
  const { seedBehaviour, iterations, prompts, base } = arg;
  // Normally, the seed behaviour implicity determines the batch size. But when we use models without seeds (like
  // ChatGPT 4o) in conjunction with the per-prompt seed behaviour, we lose out on that implicit batch size. To rectify
  // this, we need to create a batch of the right size by repeating the prompts.
  if (seedBehaviour === 'PER_PROMPT' || API_BASE_MODELS.includes(base) || VIDEO_BASE_MODELS.includes(base)) {
    return range(iterations).flatMap(() => prompts);
  }
  return prompts;
};

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
  const { state, g, base, prepend, positivePromptNode, seedNode, origin, destination } = arg;
  const { iterations, shouldRandomizeSeed, seed } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

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

  const extendedPrompts = getExtendedPrompts({ seedBehaviour, iterations, prompts, base });

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
