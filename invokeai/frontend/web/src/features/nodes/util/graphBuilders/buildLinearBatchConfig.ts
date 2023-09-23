import { NUMPY_RAND_MAX } from 'app/constants';
import { RootState } from 'app/store/store';
import { generateSeeds } from 'common/util/generateSeeds';
import { NonNullableGraph } from 'features/nodes/types/types';
import { range, unset } from 'lodash-es';
import { components } from 'services/api/schema';
import { Batch, BatchConfig, MetadataItemInvocation } from 'services/api/types';
import {
  BATCH_PROMPT,
  BATCH_SEED,
  BATCH_STYLE_PROMPT,
  CANVAS_COHERENCE_NOISE,
  METADATA_ACCUMULATOR,
  NOISE,
  POSITIVE_CONDITIONING,
} from './constants';
import {
  addBatchMetadataNodeToGraph,
  removeMetadataFromMainMetadataNode,
} from './metadata';

export const prepareLinearUIBatch = (
  state: RootState,
  graph: NonNullableGraph,
  prepend: boolean
): BatchConfig => {
  const { iterations, model, shouldRandomizeSeed, seed } = state.generation;
  const { shouldConcatSDXLStylePrompt, positiveStylePrompt } = state.sdxl;
  const { prompts, seedBehaviour } = state.dynamicPrompts;

  const data: Batch['data'] = [];

  const seedMetadataItemNode: MetadataItemInvocation = {
    id: BATCH_SEED,
    type: 'metadata_item',
    label: 'seed',
  };

  const promptMetadataItemNode: MetadataItemInvocation = {
    id: BATCH_PROMPT,
    type: 'metadata_item',
    label: 'positive_prompt',
  };

  const stylePromptMetadataItemNode: MetadataItemInvocation = {
    id: BATCH_STYLE_PROMPT,
    type: 'metadata_item',
    label: 'positive_style_prompt',
  };

  const itemNodesIds: string[] = [];

  if (prompts.length === 1) {
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

    // add to metadata
    removeMetadataFromMainMetadataNode(graph, 'seed');
    itemNodesIds.push(BATCH_SEED);
    graph.nodes[BATCH_SEED] = seedMetadataItemNode;
    zipped.push({
      node_path: BATCH_SEED,
      field_name: 'value',
      items: seeds,
    });

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

      // add to metadata
      removeMetadataFromMainMetadataNode(graph, 'seed');
      itemNodesIds.push(BATCH_SEED);
      graph.nodes[BATCH_SEED] = seedMetadataItemNode;
      firstBatchDatumList.push({
        node_path: BATCH_SEED,
        field_name: 'value',
        items: seeds,
      });

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
      removeMetadataFromMainMetadataNode(graph, 'seed');
      itemNodesIds.push(BATCH_SEED);
      graph.nodes[BATCH_SEED] = seedMetadataItemNode;
      secondBatchDatumList.push({
        node_path: BATCH_SEED,
        field_name: 'value',
        items: seeds,
      });

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

    // add to metadata
    removeMetadataFromMainMetadataNode(graph, 'positive_prompt');
    itemNodesIds.push(BATCH_PROMPT);
    graph.nodes[BATCH_PROMPT] = promptMetadataItemNode;
    firstBatchDatumList.push({
      node_path: BATCH_PROMPT,
      field_name: 'value',
      items: extendedPrompts,
    });

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

      // add to metadata
      removeMetadataFromMainMetadataNode(graph, 'positive_style_prompt');
      itemNodesIds.push(BATCH_STYLE_PROMPT);
      graph.nodes[BATCH_STYLE_PROMPT] = stylePromptMetadataItemNode;
      firstBatchDatumList.push({
        node_path: BATCH_STYLE_PROMPT,
        field_name: 'value',
        items: extendedPrompts,
      });
    }

    data.push(firstBatchDatumList);
  }

  addBatchMetadataNodeToGraph(graph, itemNodesIds);

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
