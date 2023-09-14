import { RootState } from 'app/store/store';
import { NonNullableGraph } from 'features/nodes/types/types';
import { unset } from 'lodash-es';
import { components } from 'services/api/schema';
import { BatchConfig } from 'services/api/types';
import { METADATA_ACCUMULATOR, POSITIVE_CONDITIONING } from './constants';

export const prepareLinearUIBatch = (
  state: RootState,
  graph: NonNullableGraph,
  prepend: boolean
): BatchConfig => {
  const { iterations, model } = state.generation;
  const { shouldConcatSDXLStylePrompt, positiveStylePrompt } = state.sdxl;
  const { prompts } = state.dynamicPrompts;

  const data: BatchConfig['batch']['data'] = [];

  if (prompts.length > 1) {
    unset(graph.nodes[METADATA_ACCUMULATOR], 'positive_prompt');

    const zippedPrompts: components['schemas']['BatchDatum'][] = [];
    // zipped batch of prompts
    zippedPrompts.push({
      node_path: POSITIVE_CONDITIONING,
      field_name: 'prompt',
      items: prompts,
    });

    zippedPrompts.push({
      node_path: METADATA_ACCUMULATOR,
      field_name: 'positive_prompt',
      items: prompts,
    });

    if (shouldConcatSDXLStylePrompt && model?.base_model === 'sdxl') {
      unset(graph.nodes[METADATA_ACCUMULATOR], 'positive_style_prompt');
      const stylePrompts = prompts.map((p) =>
        [p, positiveStylePrompt].join(' ')
      );
      zippedPrompts.push({
        node_path: POSITIVE_CONDITIONING,
        field_name: 'style',
        items: stylePrompts,
      });

      zippedPrompts.push({
        node_path: METADATA_ACCUMULATOR,
        field_name: 'positive_style_prompt',
        items: stylePrompts,
      });
    }

    data.push(zippedPrompts);
  }

  const enqueueBatchArg: BatchConfig = {
    prepend,
    batch: {
      graph,
      runs: iterations,
      data: data.length ? data : undefined,
    },
  };

  return enqueueBatchArg;
};
