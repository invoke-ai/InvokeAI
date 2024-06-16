import { getStore } from 'app/store/nanostores/store';
import { deepClone } from 'common/util/deepClone';
import openBase64ImageInTab from 'common/util/openBase64ImageInTab';
import { blobToDataURL } from "features/controlLayers/konva/util";
import { RG_LAYER_NAME } from 'features/controlLayers/konva/naming';
import { renderers } from 'features/controlLayers/konva/renderers/layers';
import { rgMaskImageUploaded } from 'features/controlLayers/store/canvasV2Slice';
import type { Dimensions, IPAdapterData, RegionalGuidanceData } from 'features/controlLayers/store/types';
import {
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
} from 'features/nodes/util/graph/constants';
import { addIPAdapterCollectorSafe, isValidIPAdapter } from 'features/nodes/util/graph/generation/addIPAdapters';
import type { Graph } from 'features/nodes/util/graph/generation/Graph';
import Konva from 'konva';
import type { IRect } from 'konva/lib/types';
import { size } from 'lodash-es';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { BaseModelType, ImageDTO, Invocation } from 'services/api/types';
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
  regions: RegionalGuidanceData[],
  g: Graph,
  documentSize: Dimensions,
  bbox: IRect,
  base: BaseModelType,
  denoise: Invocation<'denoise_latents'>,
  posCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  posCondCollect: Invocation<'collect'>,
  negCondCollect: Invocation<'collect'>
): Promise<RegionalGuidanceData[]> => {
  const isSDXL = base === 'sdxl';

  const validRegions = regions.filter((rg) => isValidRegion(rg, base));
  const blobs = await getRGMaskBlobs(validRegions, documentSize, bbox);
  assert(size(blobs) === size(validRegions), 'Mismatch between layer IDs and blobs');

  for (const rg of validRegions) {
    const blob = blobs[rg.id];
    assert(blob, `Blob for layer ${rg.id} not found`);
    // Upload the mask image, or get the cached image if it exists
    const { image_name } = await getMaskImage(rg, blob);

    // The main mask-to-tensor node
    const maskToTensor = g.addNode({
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${rg.id}`,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    });

    if (rg.positivePrompt) {
      // The main positive conditioning node
      const regionalPosCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${rg.id}`,
              prompt: rg.positivePrompt,
              style: rg.positivePrompt, // TODO: Should we put the positive prompt in both fields?
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${rg.id}`,
              prompt: rg.positivePrompt,
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

    if (rg.negativePrompt) {
      // The main negative conditioning node
      const regionalNegCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${rg.id}`,
              prompt: rg.negativePrompt,
              style: rg.negativePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${rg.id}`,
              prompt: rg.negativePrompt,
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
    if (rg.autoNegative === 'invert' && rg.positivePrompt) {
      // We re-use the mask image, but invert it when converting to tensor
      const invertTensorMask = g.addNode({
        id: `${PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX}_${rg.id}`,
        type: 'invert_tensor_mask',
      });
      // Connect the OG mask image to the inverted mask-to-tensor node
      g.addEdge(maskToTensor, 'mask', invertTensorMask, 'mask');
      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the positive prompt
      const regionalPosCondInverted = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${rg.id}`,
              prompt: rg.positivePrompt,
              style: rg.positivePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${rg.id}`,
              prompt: rg.positivePrompt,
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

    const validRGIPAdapters: IPAdapterData[] = rg.ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base));

    for (const ipa of validRGIPAdapters) {
      const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);
      const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipa;
      assert(model, 'IP Adapter model is required');
      assert(image, 'IP Adapter image is required');

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
          image_name: image.name,
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

export const isValidRegion = (rg: RegionalGuidanceData, base: BaseModelType) => {
  const hasTextPrompt = Boolean(rg.positivePrompt || rg.negativePrompt);
  const hasIPAdapter = rg.ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base)).length > 0;
  return hasTextPrompt || hasIPAdapter;
};

export const getMaskImage = async (rg: RegionalGuidanceData, blob: Blob): Promise<ImageDTO> => {
  const { id, imageCache } = rg;
  if (imageCache) {
    const imageDTO = await getImageDTO(imageCache.name);
    if (imageDTO) {
      return imageDTO;
    }
  }
  const { dispatch } = getStore();
  // No cached mask, or the cached image no longer exists - we need to upload the mask image
  const file = new File([blob], `${rg.id}_mask.png`, { type: 'image/png' });
  const req = dispatch(
    imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
  );
  req.reset();

  const imageDTO = await req.unwrap();
  dispatch(rgMaskImageUploaded({ id, imageDTO }));
  return imageDTO;
};

/**
 * Get the blobs of all regional prompt layers. Only visible layers are returned.
 * @param layerIds The IDs of the layers to get blobs for. If not provided, all regional prompt layers are used.
 * @param preview Whether to open a new tab displaying each layer.
 * @returns A map of layer IDs to blobs.
 */

export const getRGMaskBlobs = async (
  regions: RegionalGuidanceData[],
  documentSize: Dimensions,
  bbox: IRect,
  preview: boolean = false
): Promise<Record<string, Blob>> => {
  const container = document.createElement('div');
  const stage = new Konva.Stage({ container, ...documentSize });
  renderers.renderLayers(stage, [], [], regions, 1, 'brush', null, getImageDTO);
  const konvaLayers = stage.find<Konva.Layer>(`.${RG_LAYER_NAME}`);
  const blobs: Record<string, Blob> = {};

  // First remove all layers
  for (const layer of konvaLayers) {
    layer.remove();
  }

  // Next render each layer to a blob
  for (const layer of konvaLayers) {
    const rg = regions.find((l) => l.id === layer.id());
    if (!rg) {
      continue;
    }
    stage.add(layer);
    const blob = await new Promise<Blob>((resolve) => {
      stage.toBlob({
        callback: (blob) => {
          assert(blob, 'Blob is null');
          resolve(blob);
        },
        ...bbox,
      });
    });

    if (preview) {
      const base64 = await blobToDataURL(blob);
      openBase64ImageInTab([
        {
          base64,
          caption: `${rg.id}: ${rg.positivePrompt} / ${rg.negativePrompt}`,
        },
      ]);
    }
    layer.remove();
    blobs[layer.id()] = blob;
  }

  return blobs;
};
