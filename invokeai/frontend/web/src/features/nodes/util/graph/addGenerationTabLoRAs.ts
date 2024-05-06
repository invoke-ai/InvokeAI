import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { zModelIdentifierField } from 'features/nodes/types/common';
import { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import { filter, size } from 'lodash-es';
import type { Invocation, S } from 'services/api/types';
import { assert } from 'tsafe';

import { LORA_LOADER } from './constants';

export const addGenerationTabLoRAs = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  unetSource: Invocation<'main_model_loader'> | Invocation<'sdxl_model_loader'> | Invocation<'seamless'>,
  clipSkip: Invocation<'clip_skip'>,
  posCond: Invocation<'compel'>,
  negCond: Invocation<'compel'>
): void => {
  /**
   * LoRA nodes get the UNet and CLIP models from the main model loader and apply the LoRA to them.
   * They then output the UNet and CLIP models references on to either the next LoRA in the chain,
   * or to the inference/conditioning nodes.
   *
   * So we need to inject a LoRA chain into the graph.
   */

  const enabledLoRAs = filter(state.lora.loras, (l) => l.isEnabled ?? false);
  const loraCount = size(enabledLoRAs);

  if (loraCount === 0) {
    return;
  }

  // Remove modelLoaderNodeId unet connection to feed it to LoRAs
  console.log(deepClone(g)._graph.edges.map((e) => Graph.edgeToString(e)));
  g.deleteEdgesFrom(unetSource, 'unet');
  console.log(deepClone(g)._graph.edges.map((e) => Graph.edgeToString(e)));
  if (clipSkip) {
    // Remove CLIP_SKIP connections to conditionings to feed it through LoRAs
    g.deleteEdgesFrom(clipSkip, 'clip');
  }
  console.log(deepClone(g)._graph.edges.map((e) => Graph.edgeToString(e)));

  // we need to remember the last lora so we can chain from it
  let lastLoRALoader: Invocation<'lora_loader'> | null = null;
  let currentLoraIndex = 0;
  const loraMetadata: S['LoRAMetadataField'][] = [];

  for (const lora of enabledLoRAs) {
    const { weight } = lora;
    const { key } = lora.model;
    const currentLoraNodeId = `${LORA_LOADER}_${key}`;
    const parsedModel = zModelIdentifierField.parse(lora.model);

    const currentLoRALoader = g.addNode({
      type: 'lora_loader',
      id: currentLoraNodeId,
      lora: parsedModel,
      weight,
    });

    loraMetadata.push({
      model: parsedModel,
      weight,
    });

    // add to graph
    if (currentLoraIndex === 0) {
      // first lora = start the lora chain, attach directly to model loader
      g.addEdge(unetSource, 'unet', currentLoRALoader, 'unet');
      g.addEdge(clipSkip, 'clip', currentLoRALoader, 'clip');
    } else {
      assert(lastLoRALoader !== null);
      // we are in the middle of the lora chain, instead connect to the previous lora
      g.addEdge(lastLoRALoader, 'unet', currentLoRALoader, 'unet');
      g.addEdge(lastLoRALoader, 'clip', currentLoRALoader, 'clip');
    }

    if (currentLoraIndex === loraCount - 1) {
      // final lora, end the lora chain - we need to connect up to inference and conditioning nodes
      g.addEdge(currentLoRALoader, 'unet', denoise, 'unet');
      g.addEdge(currentLoRALoader, 'clip', posCond, 'clip');
      g.addEdge(currentLoRALoader, 'clip', negCond, 'clip');
    }

    // increment the lora for the next one in the chain
    lastLoRALoader = currentLoRALoader;
    currentLoraIndex += 1;
  }

  MetadataUtil.add(g, { loras: loraMetadata });
};
