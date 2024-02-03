import type { RootState } from 'app/store/store';
import type { LoRAMetadataItem } from 'features/nodes/types/metadata';
import { zLoRAMetadataItem } from 'features/nodes/types/metadata';
import { forEach, size } from 'lodash-es';
import type { NonNullableGraph, SDXLLoraLoaderInvocation } from 'services/api/types';

import {
  CANVAS_COHERENCE_DENOISE_LATENTS,
  LORA_LOADER,
  NEGATIVE_CONDITIONING,
  POSITIVE_CONDITIONING,
  SDXL_CANVAS_INPAINT_GRAPH,
  SDXL_CANVAS_OUTPAINT_GRAPH,
  SDXL_MODEL_LOADER,
  SDXL_REFINER_INPAINT_CREATE_MASK,
  SEAMLESS,
} from './constants';
import { upsertMetadata } from './metadata';

export const addSDXLLoRAsToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  baseNodeId: string,
  modelLoaderNodeId: string = SDXL_MODEL_LOADER
): void => {
  /**
   * LoRA nodes get the UNet and CLIP models from the main model loader and apply the LoRA to them.
   * They then output the UNet and CLIP models references on to either the next LoRA in the chain,
   * or to the inference/conditioning nodes.
   *
   * So we need to inject a LoRA chain into the graph.
   */

  const { loras } = state.lora;
  const loraCount = size(loras);

  if (loraCount === 0) {
    return;
  }

  const loraMetadata: LoRAMetadataItem[] = [];

  // Handle Seamless Plugs
  const unetLoaderId = modelLoaderNodeId;
  let clipLoaderId = modelLoaderNodeId;
  if ([SEAMLESS, SDXL_REFINER_INPAINT_CREATE_MASK].includes(modelLoaderNodeId)) {
    clipLoaderId = SDXL_MODEL_LOADER;
  }

  // Remove modelLoaderNodeId unet/clip/clip2 connections to feed it to LoRAs
  graph.edges = graph.edges.filter(
    (e) =>
      !(e.source.node_id === unetLoaderId && ['unet'].includes(e.source.field)) &&
      !(e.source.node_id === clipLoaderId && ['clip'].includes(e.source.field)) &&
      !(e.source.node_id === clipLoaderId && ['clip2'].includes(e.source.field))
  );

  // we need to remember the last lora so we can chain from it
  let lastLoraNodeId = '';
  let currentLoraIndex = 0;

  forEach(loras, (lora) => {
    const { model_name, base_model, weight } = lora;
    const currentLoraNodeId = `${LORA_LOADER}_${model_name.replace('.', '_')}`;

    const loraLoaderNode: SDXLLoraLoaderInvocation = {
      type: 'sdxl_lora_loader',
      id: currentLoraNodeId,
      is_intermediate: true,
      lora: { model_name, base_model },
      weight,
    };

    loraMetadata.push(
      zLoRAMetadataItem.parse({
        lora: { model_name, base_model },
        weight,
      })
    );

    // add to graph
    graph.nodes[currentLoraNodeId] = loraLoaderNode;
    if (currentLoraIndex === 0) {
      // first lora = start the lora chain, attach directly to model loader
      graph.edges.push({
        source: {
          node_id: unetLoaderId,
          field: 'unet',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
      });

      graph.edges.push({
        source: {
          node_id: clipLoaderId,
          field: 'clip',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: clipLoaderId,
          field: 'clip2',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
      });
    } else {
      // we are in the middle of the lora chain, instead connect to the previous lora
      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'unet',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
      });
      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: lastLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
      });
    }

    if (currentLoraIndex === loraCount - 1) {
      // final lora, end the lora chain - we need to connect up to inference and conditioning nodes
      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'unet',
        },
        destination: {
          node_id: baseNodeId,
          field: 'unet',
        },
      });

      if (graph.id && [SDXL_CANVAS_INPAINT_GRAPH, SDXL_CANVAS_OUTPAINT_GRAPH].includes(graph.id)) {
        graph.edges.push({
          source: {
            node_id: currentLoraNodeId,
            field: 'unet',
          },
          destination: {
            node_id: CANVAS_COHERENCE_DENOISE_LATENTS,
            field: 'unet',
          },
        });
      }

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: POSITIVE_CONDITIONING,
          field: 'clip2',
        },
      });

      graph.edges.push({
        source: {
          node_id: currentLoraNodeId,
          field: 'clip2',
        },
        destination: {
          node_id: NEGATIVE_CONDITIONING,
          field: 'clip2',
        },
      });
    }

    // increment the lora for the next one in the chain
    lastLoraNodeId = currentLoraNodeId;
    currentLoraIndex += 1;
  });

  upsertMetadata(graph, { loras: loraMetadata });
};
