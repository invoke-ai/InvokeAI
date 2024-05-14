import type { RootState } from 'app/store/store';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { filter, size } from 'lodash-es';
import type { Invocation, S } from 'services/api/types';

import { LORA_LOADER } from './constants';

export const addGenerationTabLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  modelLoader: Invocation<'main_model_loader'>,
  seamless: Invocation<'seamless'> | null,
  clipSkip: Invocation<'clip_skip'>,
  posCond: Invocation<'compel'>,
  negCond: Invocation<'compel'>
): void => {
  const enabledLoRAs = filter(state.lora.loras, (l) => l.isEnabled ?? false);
  const loraCount = size(enabledLoRAs);

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the UNet and CLIP.
  const loraCollector = g.addNode({
    id: `${LORA_LOADER}_collect`,
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    id: LORA_LOADER,
    type: 'lora_collection_loader',
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use seamless as UNet input if it exists, otherwise use the model loader
  g.addEdge(seamless ?? modelLoader, 'unet', loraCollectionLoader, 'unet');
  g.addEdge(clipSkip, 'clip', loraCollectionLoader, 'clip');
  // Reroute UNet & CLIP connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['unet']);
  g.deleteEdgesTo(posCond, ['clip']);
  g.deleteEdgesTo(negCond, ['clip']);
  g.addEdge(loraCollectionLoader, 'unet', denoise, 'unet');
  g.addEdge(loraCollectionLoader, 'clip', posCond, 'clip');
  g.addEdge(loraCollectionLoader, 'clip', negCond, 'clip');

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const { key } = lora.model;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const loraSelector = g.addNode({
      type: 'lora_selector',
      id: `${LORA_LOADER}_${key}`,
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
