import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type { LoRA } from 'features/controlLayers/store/types';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { Invocation, S } from 'services/api/types';

type AddSDXLLoRAsOptions = {
  loras?: LoRA[];
  metadataKey?: 'loras' | 'hrf_loras';
  idPrefix?: string;
  extraPositiveConditioning?: Invocation<'sdxl_compel_prompt'>[];
  extraNegativeConditioning?: Invocation<'sdxl_compel_prompt'>[];
};

export const addSDXLLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'> | Invocation<'tiled_multi_diffusion_denoise_latents'>,
  modelLoader: Invocation<'sdxl_model_loader'>,
  seamless: Invocation<'seamless'> | null,
  posCond: Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'sdxl_compel_prompt'>,
  options?: AddSDXLLoRAsOptions
): void => {
  const enabledLoRAs = (options?.loras ?? state.loras.loras).filter((l) => l.isEnabled && l.model.base === 'sdxl');
  const loraCount = enabledLoRAs.length;
  const positiveConditioning = [posCond, ...(options?.extraPositiveConditioning ?? [])];
  const negativeConditioning = [negCond, ...(options?.extraNegativeConditioning ?? [])];

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the UNet and CLIP.
  const loraCollector = g.addNode({
    id: getPrefixedId(options?.idPrefix ? `${options.idPrefix}_lora_collector` : 'lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'sdxl_lora_collection_loader',
    id: getPrefixedId(
      options?.idPrefix ? `${options.idPrefix}_sdxl_lora_collection_loader` : 'sdxl_lora_collection_loader'
    ),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use seamless as UNet input if it exists, otherwise use the model loader
  g.addEdge(seamless ?? modelLoader, 'unet', loraCollectionLoader, 'unet');
  g.addEdge(modelLoader, 'clip', loraCollectionLoader, 'clip');
  g.addEdge(modelLoader, 'clip2', loraCollectionLoader, 'clip2');
  // Reroute UNet & CLIP connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['unet']);
  g.addEdge(loraCollectionLoader, 'unet', denoise, 'unet');

  for (const cond of positiveConditioning) {
    g.deleteEdgesTo(cond, ['clip', 'clip2']);
    g.addEdge(loraCollectionLoader, 'clip', cond, 'clip');
    g.addEdge(loraCollectionLoader, 'clip2', cond, 'clip2');
  }

  for (const cond of negativeConditioning) {
    g.deleteEdgesTo(cond, ['clip', 'clip2']);
    g.addEdge(loraCollectionLoader, 'clip', cond, 'clip');
    g.addEdge(loraCollectionLoader, 'clip2', cond, 'clip2');
  }

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: getPrefixedId(options?.idPrefix ? `${options.idPrefix}_lora_selector` : 'lora_selector'),
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({
      model: parsedModel,
      weight,
    });

    g.addEdge(loraSelector, 'lora', loraCollector, 'item');
  }

  if (options?.metadataKey === 'hrf_loras') {
    g.upsertMetadata({ hrf_loras: loraMetadata });
  } else {
    g.upsertMetadata({ loras: loraMetadata });
  }
};
