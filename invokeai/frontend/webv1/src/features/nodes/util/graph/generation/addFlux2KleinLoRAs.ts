import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { zModelIdentifierField } from 'features/nodes/types/common';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { modelConfigsAdapterSelectors, selectModelConfigsQuery } from 'services/api/endpoints/models';
import type { Invocation, S } from 'services/api/types';

export const addFlux2KleinLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'flux2_denoise'>,
  modelLoader: Invocation<'flux2_klein_model_loader'>,
  textEncoder: Invocation<'flux2_klein_text_encoder'>
): void => {
  // Klein and dev LoRAs both carry `base === 'flux2'`; a base-only filter would wire a dev
  // LoRA (hidden 5120/6144) into the Klein graph → guaranteed shape-mismatch during denoise
  // (Klein hidden 3072/4096). The bare identifier carries no variant, so resolve each config
  // and drop dev LoRAs here. The Klein LoRA loaders reject dev LoRAs server-side too.
  const modelConfigsData = selectModelConfigsQuery(state).data;
  const isNotDevVariant = (key: string): boolean => {
    const config = modelConfigsData ? modelConfigsAdapterSelectors.selectById(modelConfigsData, key) : undefined;
    const variant = config && 'variant' in config ? config.variant : undefined;
    // Fail open when the variant can't be determined; the backend still guards.
    return variant !== 'dev';
  };
  const enabledLoRAs = state.loras.loras.filter(
    (l) => l.isEnabled && l.model.base === 'flux2' && isNotDevVariant(l.model.key)
  );
  const loraCount = enabledLoRAs.length;

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: S['LoRAMetadataField'][] = [];

  // We will collect LoRAs into a single collection node, then pass them to the LoRA collection loader, which applies
  // each LoRA to the transformer and Qwen3 text encoder.
  const loraCollector = g.addNode({
    id: getPrefixedId('lora_collector'),
    type: 'collect',
  });
  const loraCollectionLoader = g.addNode({
    type: 'flux2_klein_lora_collection_loader',
    id: getPrefixedId('flux2_klein_lora_collection_loader'),
  });

  g.addEdge(loraCollector, 'collection', loraCollectionLoader, 'loras');
  // Use model loader as transformer and qwen3_encoder input
  g.addEdge(modelLoader, 'transformer', loraCollectionLoader, 'transformer');
  g.addEdge(modelLoader, 'qwen3_encoder', loraCollectionLoader, 'qwen3_encoder');
  // Reroute model connections through the LoRA collection loader
  g.deleteEdgesTo(denoise, ['transformer']);
  g.deleteEdgesTo(textEncoder, ['qwen3_encoder']);
  g.addEdge(loraCollectionLoader, 'transformer', denoise, 'transformer');
  g.addEdge(loraCollectionLoader, 'qwen3_encoder', textEncoder, 'qwen3_encoder');

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
