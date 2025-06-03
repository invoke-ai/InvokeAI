import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import randomInt from 'common/util/randomInt';
import type { SeedBehaviour } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import type { ModelIdentifierField } from 'features/nodes/types/common';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { range } from 'lodash-es';
import type { components } from 'services/api/schema';
import type { Batch, EnqueueBatchArg } from 'services/api/types';
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
  if (seedBehaviour === 'PER_PROMPT' || model.base === 'chatgpt-4o') {
    return range(iterations).flatMap(() => prompts);
  }
  return prompts;
};

export const prepareLinearUIBatch = (arg: {
  state: RootState;
  g: Graph;
  prepend: boolean;
  seedFieldIdentifier?: FieldIdentifier;
  positivePromptFieldIdentifier: FieldIdentifier;
  origin: 'canvas' | 'workflows' | 'upscaling';
  destination: string;
}): EnqueueBatchArg => {
  const { state, g, prepend, seedFieldIdentifier, positivePromptFieldIdentifier, origin, destination } = arg;
  const { iterations, model, shouldRandomizeSeed, seed, shouldConcatPrompts } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  assert(model, 'No model found in state when preparing batch');

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

  // add seeds first to ensure the output order groups the prompts
  if (seedFieldIdentifier && seedBehaviour === 'PER_PROMPT') {
    const seeds = generateSeeds({
      count: prompts.length * iterations,
      // Imagen3's support for seeded generation is iffy, we are just not going too use it in linear UI generations.
      start:
        model.base === 'imagen3' || model.base === 'imagen4'
          ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
          : shouldRandomizeSeed
            ? undefined
            : seed,
    });

    firstBatchDatumList.push({
      node_path: seedFieldIdentifier.nodeId,
      field_name: seedFieldIdentifier.fieldName,
      items: seeds,
    });

    // add to metadata
    g.removeMetadata(['seed']);
    firstBatchDatumList.push({
      node_path: g.getMetadataNode().id,
      field_name: 'seed',
      items: seeds,
    });
  } else if (seedFieldIdentifier && seedBehaviour === 'PER_ITERATION') {
    // seedBehaviour = SeedBehaviour.PerRun
    const seeds = generateSeeds({
      count: iterations,
      // Imagen3's support for seeded generation is iffy, we are just not going too use in in linear UI generations.
      start:
        model.base === 'imagen3' || model.base === 'imagen4'
          ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX)
          : shouldRandomizeSeed
            ? undefined
            : seed,
    });

    secondBatchDatumList.push({
      node_path: seedFieldIdentifier.nodeId,
      field_name: seedFieldIdentifier.fieldName,
      items: seeds,
    });

    // add to metadata
    g.removeMetadata(['seed']);
    secondBatchDatumList.push({
      node_path: g.getMetadataNode().id,
      field_name: 'seed',
      items: seeds,
    });
    data.push(secondBatchDatumList);
  }

  const extendedPrompts = getExtendedPrompts({ seedBehaviour, iterations, prompts, model });

  // zipped batch of prompts
  firstBatchDatumList.push({
    node_path: positivePromptFieldIdentifier.nodeId,
    field_name: positivePromptFieldIdentifier.fieldName,
    items: extendedPrompts,
  });

  // add to metadata
  g.removeMetadata(['positive_prompt']);
  firstBatchDatumList.push({
    node_path: g.getMetadataNode().id,
    field_name: 'positive_prompt',
    items: extendedPrompts,
  });

  if (shouldConcatPrompts && model.base === 'sdxl') {
    firstBatchDatumList.push({
      node_path: positivePromptFieldIdentifier.nodeId,
      field_name: 'style',
      items: extendedPrompts,
    });

    // add to metadata
    g.removeMetadata(['positive_style_prompt']);
    firstBatchDatumList.push({
      node_path: g.getMetadataNode().id,
      field_name: 'positive_style_prompt',
      items: extendedPrompts,
    });
  }

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
