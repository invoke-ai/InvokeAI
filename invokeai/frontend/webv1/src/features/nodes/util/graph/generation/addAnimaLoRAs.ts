import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

export const addAnimaLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'anima_denoise'>,
  modelLoader: Invocation<'anima_model_loader'>,
  posCond: Invocation<'anima_text_encoder'>,
  negCond: Invocation<'anima_text_encoder'> | null
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'anima');
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the transformer and Qwen3 encoder.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'anima_lora_collection_loader',
    id: getPrefixedId('anima_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use model loader as transformer input
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', loraCollectionLoader, 'qwen3_encoder');
  // Reroute model connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(posCond, ['qwen3_encoder']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 'qwen3_encoder', posCond, 'qwen3_encoder');
  // Only reroute negCond if it exists (guidance_scale > 1)
  if (negCond !== null) {
    g.deleteEdgesTo(negCond, ['qwen3_encoder']);
    g.addEdge(loraCollectionLoader, 'qwen3_encoder', negCond, 'qwen3_encoder');
  }

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
