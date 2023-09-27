import { NUMPY_RAND_MAX } from 'app/constants';
import { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { NonNullableGraph } from 'features/nodes/types/types';
import { range, unset } from 'lodash-es';
import { components } from 'services/api/schema';
import { Batch, BatchConfig } from 'services/api/types';
import {
  CANVAS_COHERENCE_NOISE,
  METADATA_ACCUMULATOR,
  NOISE,
  POSITIVE_CONDITIONING,
} from './constants';

export const prepareLinearUIBatch = (
  state: RootState,
  graph: NonNullableGraph,
  prepend: boolean
): BatchConfig => {
  const { iterations, model, shouldRandomizeSeed, seed } = state.generation;
  const { shouldConcatSDXLStylePrompt, positiveStylePrompt } = state.sdxl;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const data: Batch['data'] = [];

  if (prompts.length === 1) {
    unset(graph.nodes[METADATA_ACCUMULATOR], 'seed');
    const seeds = generateSeeds({
      count: iterations,
      start: shouldRandomizeSeed ? undefined : seed,
    });

    const zipped: components['schemas']['BatchDatum'][] = [];

    if (graph.nodes[NOISE]) {
      zipped.push({
        node_path: NOISE,
        field_name: 'seed',
        items: seeds,
      });
    }

    if (graph.nodes[METADATA_ACCUMULATOR]) {
      console.log(
        'adding seed to metadata accumulator',
        METADATA_ACCUMULATOR,
        seeds,
        '--',
        graph.nodes[METADATA_ACCUMULATOR]
      );
      zipped.push({
        node_path: METADATA_ACCUMULATOR,
        field_name: 'seed',
        items: seeds,
      });
    }

    if (graph.nodes[CANVAS_COHERENCE_NOISE]) {
      zipped.push({
        node_path: CANVAS_COHERENCE_NOISE,
        field_name: 'seed',
        items: seeds.map((seed) => (seed + 1) % NUMPY_RAND_MAX),
      });
    }

    data.push(zipped);
  } else {
    // prompts.length > 1 aka dynamic prompts
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

      if (graph.nodes[METADATA_ACCUMULATOR]) {
        firstBatchDatumList.push({
          node_path: METADATA_ACCUMULATOR,
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
      if (graph.nodes[METADATA_ACCUMULATOR]) {
        secondBatchDatumList.push({
          node_path: METADATA_ACCUMULATOR,
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

    const extendedPrompts =
      seedBehaviour === 'PER_PROMPT'
        ? range(iterations).flatMap(() => prompts)
        : prompts;

    // zipped batch of prompts
    if (graph.nodes[POSITIVE_CONDITIONING]) {
      firstBatchDatumList.push({
        node_path: POSITIVE_CONDITIONING,
        field_name: 'prompt',
        items: extendedPrompts,
      });
    }

    if (graph.nodes[METADATA_ACCUMULATOR]) {
      console.log(
        'adding prompt to metadata accumulator',
        METADATA_ACCUMULATOR
      );
      firstBatchDatumList.push({
        node_path: METADATA_ACCUMULATOR,
        field_name: 'positive_prompt',
        items: extendedPrompts,
      });
    }

    if (shouldConcatSDXLStylePrompt && model?.base_model === 'sdxl') {
      unset(graph.nodes[METADATA_ACCUMULATOR], 'positive_style_prompt');

      const stylePrompts = extendedPrompts.map((p) =>
        [p, positiveStylePrompt].join(' ')
      );

      if (graph.nodes[POSITIVE_CONDITIONING]) {
        firstBatchDatumList.push({
          node_path: POSITIVE_CONDITIONING,
          field_name: 'style',
          items: stylePrompts,
        });
      }

      if (graph.nodes[METADATA_ACCUMULATOR]) {
        console.log(
          'adding style prompt to metadata accumulator',
          METADATA_ACCUMULATOR
        );
        firstBatchDatumList.push({
          node_path: METADATA_ACCUMULATOR,
          field_name: 'positive_style_prompt',
          items: stylePrompts,
        });
      }
    }

    data.push(firstBatchDatumList);
  }

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
