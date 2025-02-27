import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

export const addSDXLLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  modelLoader: Invocation<'sdxl_model_loader'>,
  seamless: Invocation<'seamless'> | null,
  posCond: Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'sdxl_compel_prompt'>
): void => {
  const enabledLoRAs = state.loras.loras.filter((l) => l.isEnabled && l.model.base === 'sdxl');
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the UNet and CLIP.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'sdxl_lora_collection_loader',
    id: getPrefixedId('sdxl_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use seamless as UNet input if it exists, otherwise use the model loader
  g.addEdge(seamless ?? modelLoader, 'unet', loraCollectionLoader, 'unet');
  g.addEdge(modelLoader, 'clip', loraCollectionLoader, 'clip');
  g.addEdge(modelLoader, 'clip2', loraCollectionLoader, 'clip2');
  // Reroute UNet & CLIP connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['unet']);
  g.deleteEdgesTo(posCond, ['clip', 'clip2']);
  g.deleteEdgesTo(negCond, ['clip', 'clip2']);
  g.addEdge(loraCollectionLoader, 'unet', denoise, 'unet');
  g.addEdge(loraCollectionLoader, 'clip', posCond, 'clip');
  g.addEdge(loraCollectionLoader, 'clip', negCond, 'clip');
  g.addEdge(loraCollectionLoader, 'clip2', posCond, 'clip2');
  g.addEdge(loraCollectionLoader, 'clip2', negCond, 'clip2');

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
