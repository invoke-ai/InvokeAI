import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

export const addFLUXLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'flux_denoise'>,
  modelLoader: Invocation<'flux_model_loader'>,
  fluxTextEncoder: Invocation<'flux_text_encoder'>
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'flux');
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the transformer and text encoders.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'flux_lora_collection_loader',
    id: getPrefixedId('flux_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use model loader as transformer input
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 'clip', loraCollectionLoader, 'clip');
  // Reroute model connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(fluxTextEncoder, ['clip']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 'clip', fluxTextEncoder, 'clip');

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
