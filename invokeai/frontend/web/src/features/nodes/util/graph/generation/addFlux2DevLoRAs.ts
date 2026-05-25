import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

/**
 * Wire any enabled FLUX.2 LoRAs through a `flux2_dev_lora_collection_loader`,
 * patching both the transformer and the Mistral text encoder.
 */
export const addFlux2DevLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'flux2_denoise'>,
  modelLoader: Invocation<'flux2_dev_model_loader'>,
  textEncoder: Invocation<'flux2_dev_text_encoder'>
): void => {
  // Currently all `flux2` LoRAs share a single base value (the variant guard happens
  // server-side in the dev LoRA loader, which warns on mismatches).
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'flux2');
  if (enabledLoRAs.length === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'flux2_dev_lora_collection_loader',
    id: getPrefixedId('flux2_dev_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 'mistral_encoder', loraCollectionLoader, 'mistral_encoder');
  // Reroute the patched outputs back into the denoise / text encoder.
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(textEncoder, ['mistral_encoder']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 'mistral_encoder', textEncoder, 'mistral_encoder');

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: getPrefixedId('lora_selector'),
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({ model: parsedModel, weight });

    g.addEdge(loraSelector, 'lora', loraCollector, 'item');
  }

  g.upsertMetadata({ loras: loraMetadata });
};
