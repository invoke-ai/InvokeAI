import { NUMPY_RAND_MAX } from 'app/constants';
import type { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { range, some } from 'lodash-es';
import type { components } from 'services/api/schema';
import type { Batch, BatchConfig, NonNullableGraph } from 'services/api/types';

import {
  CANVAS_COHERENCE_NOISE,
  METADATA,
  NOISE,
  POSITIVE_CONDITIONING,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
} from './constants';
import { getHasMetadata, removeMetadata } from './metadata';

export const prepareLinearUIBatch = (state: RootState, graph: NonNullableGraph, prepend: boolean): BatchConfig => {
  const { iterations, model, shouldRandomizeSeed, seed } = state.generation;
  const { shouldConcatSDXLStylePrompt } = state.sdxl;
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

    if (graph.nodes[NOISE]) {
      firstBatchDatumList.push({
        node_path: NOISE,
        field_name: 'seed',
        items: seeds,
      });
    }

    // add to metadata
    if (getHasMetadata(graph)) {
      removeMetadata(graph, 'seed');
      firstBatchDatumList.push({
        node_path: METADATA,
        field_name: 'seed',
        items: seeds,
      });
    }

    if (graph.nodes[CANVAS_COHERENCE_NOISE]) {
      firstBatchDatumList.push({
        node_path: CANVAS_COHERENCE_NOISE,
        field_name: 'seed',
        items: seeds.map((seed) => (seed + 1) % NUMPY_RAND_MAX),
      });
    }
  } else {
    // seedBehaviour = SeedBehaviour.PerRun
    const seeds = generateSeeds({
      count: iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    if (graph.nodes[NOISE]) {
      secondBatchDatumList.push({
        node_path: NOISE,
        field_name: 'seed',
        items: seeds,
      });
    }

    // add to metadata
    if (getHasMetadata(graph)) {
      removeMetadata(graph, 'seed');
      secondBatchDatumList.push({
        node_path: METADATA,
        field_name: 'seed',
        items: seeds,
      });
    }

    if (graph.nodes[CANVAS_COHERENCE_NOISE]) {
      secondBatchDatumList.push({
        node_path: CANVAS_COHERENCE_NOISE,
        field_name: 'seed',
        items: seeds.map((seed) => (seed + 1) % NUMPY_RAND_MAX),
      });
    }
    data.push(secondBatchDatumList);
  }

  const extendedPrompts = seedBehaviour === 'PER_PROMPT' ? range(iterations).flatMap(() => prompts) : prompts;

  const hasRegionalPrompts = some(graph.nodes, (n) => n.id.startsWith(PROMPT_REGION_MASK_TO_TENSOR_PREFIX));

  if (!hasRegionalPrompts) {
    // zipped batch of prompts
    if (graph.nodes[POSITIVE_CONDITIONING]) {
      firstBatchDatumList.push({
        node_path: POSITIVE_CONDITIONING,
        field_name: 'prompt',
        items: extendedPrompts,
      });
    }

    // add to metadata
    if (getHasMetadata(graph)) {
      removeMetadata(graph, 'positive_prompt');
      firstBatchDatumList.push({
        node_path: METADATA,
        field_name: 'positive_prompt',
        items: extendedPrompts,
      });
    }
  }

  if (shouldConcatSDXLStylePrompt && model?.base === 'sdxl') {
    if (graph.nodes[POSITIVE_CONDITIONING]) {
      firstBatchDatumList.push({
        node_path: POSITIVE_CONDITIONING,
        field_name: 'style',
        items: extendedPrompts,
      });
    }

    // add to metadata
    if (getHasMetadata(graph)) {
      removeMetadata(graph, 'positive_style_prompt');
      firstBatchDatumList.push({
        node_path: METADATA,
        field_name: 'positive_style_prompt',
        items: extendedPrompts,
      });
    }
  }

  data.push(firstBatchDatumList);

  const enqueueBatchArg: BatchConfig = {
    prepend,
    batch: {
      graph,
      runs: 1,
      data,
    },
  };

  return enqueueBatchArg;
};
