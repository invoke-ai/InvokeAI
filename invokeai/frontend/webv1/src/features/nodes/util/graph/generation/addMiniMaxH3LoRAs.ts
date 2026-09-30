import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

/**
 * Add MiniMax H3 LoRA wiring to the graph between the model loader and the
 * denoise node.
 *
 * Each enabled H3 LoRA becomes a ``lora_selector`` feeding a ``collect``
 * node, which fans into a ``minimax_h3_lora_collection_loader`` that rewrites
 * the model loader's transformer output with the ``loras`` list populated.
 * There is no variant/expert routing (H3 has a single fl2va transformer);
 * the backend loader re-validates every LoRA's base server-side.
 */
export const addMiniMaxH3LoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'minimax_h3_denoise'>,
  modelLoader: Invocation<'minimax_h3_model_loader'>
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'minimax-h3');

  if (enabledLoRAs.length === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'minimax_h3_lora_collection_loader',
    id: getPrefixedId('minimax_h3_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.deleteEdgesTo(denoise, ['transformer']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: getPrefixedId('lora_selector'),
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({
      model: parsedModel,
      weight,
    });

    g.addEdge(loraSelector, 'lora', loraCollector, 'item');
  }

  g.upsertMetadata({ loras: loraMetadata });
};
