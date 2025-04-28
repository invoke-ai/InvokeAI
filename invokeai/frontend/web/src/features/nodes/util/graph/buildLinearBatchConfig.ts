import { NUMPY_RAND_MAX, NUMPY_RAND_MIN } from 'app/constants';
import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import randomInt from 'common/util/randomInt';
import type { FieldIdentifier } from 'features/nodes/types/field';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { range } from 'lodash-es';
import type { components } from 'services/api/schema';
import type { Batch, EnqueueBatchArg } from 'services/api/types';

export const prepareLinearUIBatch = (
  state: RootState,
  g: Graph,
  prepend: boolean,
  seedFieldIdentifier: FieldIdentifier,
  positivePromptFieldIdentifier: FieldIdentifier,
  origin: 'canvas' | 'workflows' | 'upscaling',
  destination: 'canvas' | 'gallery'
): EnqueueBatchArg => {
  const { iterations, model, shouldRandomizeSeed, seed, shouldConcatPrompts } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

  // add seeds first to ensure the output order groups the prompts
  if (seedBehaviour === 'PER_PROMPT') {
    const seeds = generateSeeds({
      count: prompts.length * iterations,
      // Imagen3's support for seeded generation is iffy, we are just not going too use in in linear UI generations.
      start:
        model?.base === 'imagen3' ? randomInt(NUMPY_RAND_MIN, NUMPY_RAND_MAX) : shouldRandomizeSeed ? undefined : seed,
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
  } else {
    // seedBehaviour = SeedBehaviour.PerRun
    const seeds = generateSeeds({
      count: iterations,
      start: shouldRandomizeSeed ? undefined : seed,
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

  const extendedPrompts = seedBehaviour === 'PER_PROMPT' ? range(iterations).flatMap(() => prompts) : prompts;

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

  if (shouldConcatPrompts && model?.base === 'sdxl') {
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
