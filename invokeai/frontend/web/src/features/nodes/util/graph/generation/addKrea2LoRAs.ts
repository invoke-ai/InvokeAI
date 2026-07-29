import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

export const addKrea2LoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'krea2_denoise'>,
  modelLoader: Invocation<'krea2_model_loader'>,
  posCond: Invocation<'krea2_text_encoder'>,
  negCond: Invocation<'krea2_text_encoder'> | null
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'krea-2');
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // Collect LoRAs into a collection node, then apply them all via the collection loader, which reroutes
  // the transformer and Qwen3-VL encoder through itself.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'krea2_lora_collection_loader',
    id: getPrefixedId('krea2_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 'qwen3_vl_encoder', loraCollectionLoader, 'qwen3_vl_encoder');
  // Reroute model connections through the LoRA collection loader.
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(posCond, ['qwen3_vl_encoder']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 'qwen3_vl_encoder', posCond, 'qwen3_vl_encoder');
  if (negCond !== null) {
    g.deleteEdgesTo(negCond, ['qwen3_vl_encoder']);
    g.addEdge(loraCollectionLoader, 'qwen3_vl_encoder', negCond, 'qwen3_vl_encoder');
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
