import { deepClone } from 'common/util/deepClone';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { CanvasIPAdapterState, Rect, CanvasRegionalGuidanceState } from 'features/controlLayers/store/types';
import {
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
} from 'features/nodes/util/graph/constants';
import { addIPAdapterCollectorSafe, isValidIPAdapter } from 'features/nodes/util/graph/generation/addIPAdapters';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

/**
 * Adds regional guidance to the graph
 * @param regions Array of regions to add
 * @param g The graph to add the layers to
 * @param base The base model type
 * @param denoise The main denoise node
 * @param posCond The positive conditioning node
 * @param negCond The negative conditioning node
 * @param posCondCollect The positive conditioning collector
 * @param negCondCollect The negative conditioning collector
 * @returns A promise that resolves to the regions that were successfully added to the graph
 */

export const addRegions = async (
  manager: CanvasManager,
  regions: CanvasRegionalGuidanceState[],
  g: Graph,
  bbox: Rect,
  base: BaseModelType,
  denoise: Invocation<'denoise_latents'>,
  posCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  posCondCollect: Invocation<'collect'>,
  negCondCollect: Invocation<'collect'>
): Promise<CanvasRegionalGuidanceState[]> => {
  const isSDXL = base === 'sdxl';

  const validRegions = regions.filter((rg) => isValidRegion(rg, base));

  for (const region of validRegions) {
    // Upload the mask image, or get the cached image if it exists
    const { image_name } = await manager.getRegionMaskImage({ id: region.id, bbox });

    // The main mask-to-tensor node
    const maskToTensor = g.addNode({
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${region.id}`,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    });

    if (region.positivePrompt) {
      // The main positive conditioning node
      const regionalPosCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${region.id}`,
              prompt: region.positivePrompt,
              style: region.positivePrompt, // TODO: Should we put the positive prompt in both fields?
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${region.id}`,
              prompt: region.positivePrompt,
            }
      );
      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', regionalPosCond, 'mask');
      // Connect the conditioning to the collector
      g.addEdge(regionalPosCond, 'conditioning', posCondCollect, 'item');
      // Copy the connections to the "global" positive conditioning node to the regional cond
      if (posCond.type === 'compel') {
        for (const edge of g.getEdgesTo(posCond, ['clip', 'mask'])) {
          // Clone the edge, but change the destination node to the regional conditioning node
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCond.id;
          g.addEdgeFromObj(clone);
        }
      } else {
        for (const edge of g.getEdgesTo(posCond, ['clip', 'clip2', 'mask'])) {
          // Clone the edge, but change the destination node to the regional conditioning node
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCond.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    if (region.negativePrompt) {
      // The main negative conditioning node
      const regionalNegCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${region.id}`,
              prompt: region.negativePrompt,
              style: region.negativePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${region.id}`,
              prompt: region.negativePrompt,
            }
      );
      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', regionalNegCond, 'mask');
      // Connect the conditioning to the collector
      g.addEdge(regionalNegCond, 'conditioning', negCondCollect, 'item');
      // Copy the connections to the "global" negative conditioning node to the regional cond
      if (negCond.type === 'compel') {
        for (const edge of g.getEdgesTo(negCond, ['clip', 'mask'])) {
          const clone = deepClone(edge);
          clone.destination.node_id = regionalNegCond.id;
          g.addEdgeFromObj(clone);
        }
      } else {
        for (const edge of g.getEdgesTo(negCond, ['clip', 'clip2', 'mask'])) {
          const clone = deepClone(edge);
          clone.destination.node_id = regionalNegCond.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    // If we are using the "invert" auto-negative setting, we need to add an additional negative conditioning node
    if (region.autoNegative === 'invert' && region.positivePrompt) {
      // We re-use the mask image, but invert it when converting to tensor
      const invertTensorMask = g.addNode({
        id: `${PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX}_${region.id}`,
        type: 'invert_tensor_mask',
      });
      // Connect the OG mask image to the inverted mask-to-tensor node
      g.addEdge(maskToTensor, 'mask', invertTensorMask, 'mask');
      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the positive prompt
      const regionalPosCondInverted = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${region.id}`,
              prompt: region.positivePrompt,
              style: region.positivePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${region.id}`,
              prompt: region.positivePrompt,
            }
      );
      // Connect the inverted mask to the conditioning
      g.addEdge(invertTensorMask, 'mask', regionalPosCondInverted, 'mask');
      // Connect the conditioning to the negative collector
      g.addEdge(regionalPosCondInverted, 'conditioning', negCondCollect, 'item');
      // Copy the connections to the "global" positive conditioning node to our regional node
      if (posCond.type === 'compel') {
        for (const edge of g.getEdgesTo(posCond, ['clip', 'mask'])) {
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCondInverted.id;
          g.addEdgeFromObj(clone);
        }
      } else {
        for (const edge of g.getEdgesTo(posCond, ['clip', 'clip2', 'mask'])) {
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCondInverted.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    const validRGIPAdapters: CanvasIPAdapterState[] = region.ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base));

    for (const ipa of validRGIPAdapters) {
      const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);
      const { id, weight, model, clipVisionModel, method, beginEndStepPct, imageObject } = ipa;
      assert(model, 'IP Adapter model is required');
      assert(imageObject, 'IP Adapter image is required');

      const ipAdapter = g.addNode({
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        weight,
        method,
        ip_adapter_model: model,
        clip_vision_model: clipVisionModel,
        begin_step_percent: beginEndStepPct[0],
        end_step_percent: beginEndStepPct[1],
        image: {
          image_name: imageObject.image.name,
        },
      });

      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', ipAdapter, 'mask');
      g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollect, 'item');
    }
  }

  g.upsertMetadata({ regions: validRegions });
  return validRegions;
};

export const isValidRegion = (rg: CanvasRegionalGuidanceState, base: BaseModelType) => {
  const hasTextPrompt = Boolean(rg.positivePrompt || rg.negativePrompt);
  const hasIPAdapter = rg.ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base)).length > 0;
  return hasTextPrompt || hasIPAdapter;
};
