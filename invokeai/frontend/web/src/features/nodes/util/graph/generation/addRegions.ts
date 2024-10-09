import { logger } from 'app/logging/logger';
import { deepClone } from 'common/util/deepClone';
import { withResultAsync } from 'common/util/result';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import type {
  CanvasRegionalGuidanceState,
  IPAdapterConfig,
  Rect,
  RegionalGuidanceReferenceImageState,
} from 'features/controlLayers/store/types';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import { serializeError } from 'serialize-error';
import type { BaseModelType, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

const log = logger('system');

type AddedRegionResult = {
  addedPositivePrompt: boolean;
  addedNegativePrompt: boolean;
  addedAutoNegativePositivePrompt: boolean;
  addedIPAdapters: number;
};

const isValidRegion = (rg: CanvasRegionalGuidanceState, base: BaseModelType) => {
  const isEnabled = rg.isEnabled;
  const hasTextPrompt = Boolean(rg.positivePrompt || rg.negativePrompt);
  const hasIPAdapter = rg.referenceImages.filter(({ ipAdapter }) => isValidIPAdapter(ipAdapter, base)).length > 0;
  return isEnabled && (hasTextPrompt || hasIPAdapter);
};

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
 * @param ipAdapterCollect The IP adapter collector
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
  negCondCollect: Invocation<'collect'>,
  ipAdapterCollect: Invocation<'collect'>
): Promise<AddedRegionResult[]> => {
  const isSDXL = base === 'sdxl';

  const validRegions = regions.filter((rg) => isValidRegion(rg, base));
  const results: AddedRegionResult[] = [];

  for (const region of validRegions) {
    const result: AddedRegionResult = {
      addedPositivePrompt: false,
      addedNegativePrompt: false,
      addedAutoNegativePositivePrompt: false,
      addedIPAdapters: 0,
    };

    const getImageDTOResult = await withResultAsync(() => {
      const adapter = manager.adapters.regionMasks.get(region.id);
      assert(adapter, 'Adapter not found');
      return adapter.renderer.rasterize({ rect: bbox, attrs: { opacity: 1, filters: [] } });
    });
    if (getImageDTOResult.isErr()) {
      log.warn({ error: serializeError(getImageDTOResult.error) }, 'Error rasterizing region mask');
      continue;
    }

    const imageDTO = getImageDTOResult.value;

    // The main mask-to-tensor node
    const maskToTensor = g.addNode({
      type: 'alpha_mask_to_tensor',
      id: getPrefixedId('prompt_region_mask_to_tensor'),
      image: {
        image_name: imageDTO.image_name,
      },
    });

    if (region.positivePrompt) {
      // The main positive conditioning node
      result.addedPositivePrompt = true;
      const regionalPosCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: getPrefixedId('prompt_region_positive_cond'),
              prompt: region.positivePrompt,
              style: region.positivePrompt, // TODO: Should we put the positive prompt in both fields?
            }
          : {
              type: 'compel',
              id: getPrefixedId('prompt_region_positive_cond'),
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
      result.addedNegativePrompt = true;
      // The main negative conditioning node
      const regionalNegCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: getPrefixedId('prompt_region_negative_cond'),
              prompt: region.negativePrompt,
              style: region.negativePrompt,
            }
          : {
              type: 'compel',
              id: getPrefixedId('prompt_region_negative_cond'),
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
    if (region.autoNegative && region.positivePrompt) {
      result.addedAutoNegativePositivePrompt = true;
      // We re-use the mask image, but invert it when converting to tensor
      const invertTensorMask = g.addNode({
        id: getPrefixedId('prompt_region_invert_tensor_mask'),
        type: 'invert_tensor_mask',
      });
      // Connect the OG mask image to the inverted mask-to-tensor node
      g.addEdge(maskToTensor, 'mask', invertTensorMask, 'mask');
      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the positive prompt
      const regionalPosCondInverted = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: getPrefixedId('prompt_region_positive_cond_inverted'),
              prompt: region.positivePrompt,
              style: region.positivePrompt,
            }
          : {
              type: 'compel',
              id: getPrefixedId('prompt_region_positive_cond_inverted'),
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

    const validRGIPAdapters: RegionalGuidanceReferenceImageState[] = region.referenceImages.filter(({ ipAdapter }) =>
      isValidIPAdapter(ipAdapter, base)
    );

    for (const { id, ipAdapter } of validRGIPAdapters) {
      result.addedIPAdapters++;
      const { weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;
      assert(model, 'IP Adapter model is required');
      assert(image, 'IP Adapter image is required');

      const ipAdapterNode = g.addNode({
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        weight,
        method,
        ip_adapter_model: model,
        clip_vision_model: clipVisionModel,
        begin_step_percent: beginEndStepPct[0],
        end_step_percent: beginEndStepPct[1],
        image: {
          image_name: image.image_name,
        },
      });

      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', ipAdapterNode, 'mask');
      g.addEdge(ipAdapterNode, 'ip_adapter', ipAdapterCollect, 'item');
    }

    results.push(result);
  }

  g.upsertMetadata({ regions: validRegions });

  return results;
};

const isValidIPAdapter = (ipAdapter: IPAdapterConfig, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ipAdapter.model);
  const modelMatchesBase = ipAdapter.model?.base === base;
  const hasImage = Boolean(ipAdapter.image);
  return hasModel && modelMatchesBase && hasImage;
};
