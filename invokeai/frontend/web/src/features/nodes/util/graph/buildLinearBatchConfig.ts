import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { range } from 'lodash-es';
import type { components } from 'services/api/schema';
import type { Batch, BatchConfig, Invocation } from 'services/api/types';

export const prepareLinearUIBatch = (
  state: RootState,
  g: Graph,
  prepend: boolean,
  noise: Invocation<'noise' | 'flux_denoise'>,
  posCond: Invocation<'compel' | 'sdxl_compel_prompt' | 'flux_text_encoder'>,
  origin: 'canvas' | 'workflows' | 'upscaling',
  destination: 'canvas' | 'gallery'
): BatchConfig => {
  const { iterations, model, shouldRandomizeSeed, seed, shouldConcatPrompts } = state.params;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const data: Batch['data'] = [];
  const firstBatchDatumList: components['schemas']['BatchDatum'][] = [];
  const secondBatchDatumList: components['schemas']['BatchDatum'][] = [];

  // add seeds first to ensure the output order groups the prompts
  if (seedBehaviour === 'PER_PROMPT') {
    const seeds = generateSeeds({
      count: prompts.length * iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    firstBatchDatumList.push({
      node_path: noise.id,
      field_name: 'seed',
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
      node_path: noise.id,
      field_name: 'seed',
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
    node_path: posCond.id,
    field_name: 'prompt',
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
      node_path: posCond.id,
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

  const enqueueBatchArg: BatchConfig = {
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
