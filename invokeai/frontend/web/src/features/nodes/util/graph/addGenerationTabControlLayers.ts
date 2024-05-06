import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  isControlAdapterLayer,
  isIPAdapterLayer,
  isRegionalGuidanceLayer,
  rgLayerMaskImageUploaded,
} from 'features/controlLayers/store/controlLayersSlice';
import type { RegionalGuidanceLayer } from 'features/controlLayers/store/types';
import {
  type ControlNetConfigV2,
  type ImageWithDims,
  type IPAdapterConfigV2,
  isControlNetConfigV2,
  isT2IAdapterConfigV2,
  type ProcessorConfig,
  type T2IAdapterConfigV2,
} from 'features/controlLayers/util/controlAdapters';
import { getRegionalPromptLayerBlobs } from 'features/controlLayers/util/getLayerBlobs';
import type { ImageField } from 'features/nodes/types/common';
import {
  CONTROL_NET_COLLECT,
  IP_ADAPTER_COLLECT,
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
  T2I_ADAPTER_COLLECT,
} from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import { size } from 'lodash-es';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO, Invocation, S } from 'services/api/types';
import { assert } from 'tsafe';

const buildControlImage = (
  image: ImageWithDims | null,
  processedImage: ImageWithDims | null,
  processorConfig: ProcessorConfig | null
): ImageField => {
  if (processedImage && processorConfig) {
    // We've processed the image in the app - use it for the control image.
    return {
      image_name: processedImage.imageName,
    };
  } else if (image) {
    // No processor selected, and we have an image - the user provided a processed image, use it for the control image.
    return {
      image_name: image.imageName,
    };
  }
  assert(false, 'Attempted to add unprocessed control image');
};

const buildControlNetMetadata = (controlNet: ControlNetConfigV2): S['ControlNetMetadataField'] => {
  const { beginEndStepPct, controlMode, image, model, processedImage, processorConfig, weight } = controlNet;

  assert(model, 'ControlNet model is required');
  assert(image, 'ControlNet image is required');

  const processed_image =
    processedImage && processorConfig
      ? {
          image_name: processedImage.imageName,
        }
      : null;

  return {
    control_model: model,
    control_weight: weight,
    control_mode: controlMode,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    image: {
      image_name: image.imageName,
    },
    processed_image,
  };
};

const addControlNetCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // Attempt to retrieve the collector
    const controlNetCollect = g.getNode(CONTROL_NET_COLLECT);
    assert(controlNetCollect.type === 'collect');
    return controlNetCollect;
  } catch {
    // Add the ControlNet collector
    const controlNetCollect = g.addNode({
      id: CONTROL_NET_COLLECT,
      type: 'collect',
    });
    g.addEdge(controlNetCollect, 'collection', denoise, 'control');
    return controlNetCollect;
  }
};

const addGlobalControlNetsToGraph = (
  controlNetConfigs: ControlNetConfigV2[],
  g: Graph,
  denoise: Invocation<'denoise_latents'>
): void => {
  if (controlNetConfigs.length === 0) {
    return;
  }
  const controlNetMetadata: S['ControlNetMetadataField'][] = [];
  const controlNetCollect = addControlNetCollectorSafe(g, denoise);

  for (const controlNetConfig of controlNetConfigs) {
    if (!controlNetConfig.model) {
      return;
    }
    const { id, beginEndStepPct, controlMode, image, model, processedImage, processorConfig, weight } =
      controlNetConfig;

    const controlNet = g.addNode({
      id: `control_net_${id}`,
      type: 'controlnet',
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      control_mode: controlMode,
      resize_mode: 'just_resize',
      control_model: model,
      control_weight: weight,
      image: buildControlImage(image, processedImage, processorConfig),
    });

    g.addEdge(controlNet, 'control', controlNetCollect, 'item');

    controlNetMetadata.push(buildControlNetMetadata(controlNetConfig));
  }
  MetadataUtil.add(g, { controlnets: controlNetMetadata });
};

const buildT2IAdapterMetadata = (t2iAdapter: T2IAdapterConfigV2): S['T2IAdapterMetadataField'] => {
  const { beginEndStepPct, image, model, processedImage, processorConfig, weight } = t2iAdapter;

  assert(model, 'T2I Adapter model is required');
  assert(image, 'T2I Adapter image is required');

  const processed_image =
    processedImage && processorConfig
      ? {
          image_name: processedImage.imageName,
        }
      : null;

  return {
    t2i_adapter_model: model,
    weight,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    image: {
      image_name: image.imageName,
    },
    processed_image,
  };
};

const addT2IAdapterCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // You see, we've already got one!
    const t2iAdapterCollect = g.getNode(T2I_ADAPTER_COLLECT);
    assert(t2iAdapterCollect.type === 'collect');
    return t2iAdapterCollect;
  } catch {
    const t2iAdapterCollect = g.addNode({
      id: T2I_ADAPTER_COLLECT,
      type: 'collect',
    });

    g.addEdge(t2iAdapterCollect, 'collection', denoise, 't2i_adapter');

    return t2iAdapterCollect;
  }
};

const addGlobalT2IAdaptersToGraph = (
  t2iAdapterConfigs: T2IAdapterConfigV2[],
  g: Graph,
  denoise: Invocation<'denoise_latents'>
): void => {
  if (t2iAdapterConfigs.length === 0) {
    return;
  }
  const t2iAdapterMetadata: S['T2IAdapterMetadataField'][] = [];
  const t2iAdapterCollect = addT2IAdapterCollectorSafe(g, denoise);

  for (const t2iAdapterConfig of t2iAdapterConfigs) {
    if (!t2iAdapterConfig.model) {
      return;
    }
    const { id, beginEndStepPct, image, model, processedImage, processorConfig, weight } = t2iAdapterConfig;

    const t2iAdapter = g.addNode({
      id: `t2i_adapter_${id}`,
      type: 't2i_adapter',
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      resize_mode: 'just_resize',
      t2i_adapter_model: model,
      weight: weight,
      image: buildControlImage(image, processedImage, processorConfig),
    });

    g.addEdge(t2iAdapter, 't2i_adapter', t2iAdapterCollect, 'item');

    t2iAdapterMetadata.push(buildT2IAdapterMetadata(t2iAdapterConfig));
  }

  MetadataUtil.add(g, { t2iAdapters: t2iAdapterMetadata });
};

const buildIPAdapterMetadata = (ipAdapter: IPAdapterConfigV2): S['IPAdapterMetadataField'] => {
  const { weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;

  assert(model, 'IP Adapter model is required');
  assert(image, 'IP Adapter image is required');

  return {
    ip_adapter_model: model,
    clip_vision_model: clipVisionModel,
    weight,
    method,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    image: {
      image_name: image.imageName,
    },
  };
};

const addIPAdapterCollectorSafe = (g: Graph, denoise: Invocation<'denoise_latents'>): Invocation<'collect'> => {
  try {
    // You see, we've already got one!
    const ipAdapterCollect = g.getNode(IP_ADAPTER_COLLECT);
    assert(ipAdapterCollect.type === 'collect');
    return ipAdapterCollect;
  } catch {
    const ipAdapterCollect = g.addNode({
      id: IP_ADAPTER_COLLECT,
      type: 'collect',
    });
    g.addEdge(ipAdapterCollect, 'collection', denoise, 'ip_adapter');
    return ipAdapterCollect;
  }
};

const addGlobalIPAdaptersToGraph = (
  ipAdapterConfigs: IPAdapterConfigV2[],
  g: Graph,
  denoise: Invocation<'denoise_latents'>
): void => {
  if (ipAdapterConfigs.length === 0) {
    return;
  }
  const ipAdapterMetdata: S['IPAdapterMetadataField'][] = [];
  const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);

  for (const ipAdapterConfig of ipAdapterConfigs) {
    const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapterConfig;
    assert(image, 'IP Adapter image is required');
    assert(model, 'IP Adapter model is required');

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
        image_name: image.imageName,
      },
    });
    g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollect, 'item');
    ipAdapterMetdata.push(buildIPAdapterMetadata(ipAdapterConfig));
  }

  MetadataUtil.add(g, { ipAdapters: ipAdapterMetdata });
};

export const addGenerationTabControlLayers = async (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  posCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  posCondCollect: Invocation<'collect'>,
  negCondCollect: Invocation<'collect'>
) => {
  const mainModel = state.generation.model;
  assert(mainModel, 'Missing main model when building graph');
  const isSDXL = mainModel.base === 'sdxl';

  // Add global control adapters
  const globalControlNetConfigs = state.controlLayers.present.layers
    // Must be a CA layer
    .filter(isControlAdapterLayer)
    // Must be enabled
    .filter((l) => l.isEnabled)
    // We want the CAs themselves
    .map((l) => l.controlAdapter)
    // Must be a ControlNet
    .filter(isControlNetConfigV2)
    .filter((ca) => {
      const hasModel = Boolean(ca.model);
      const modelMatchesBase = ca.model?.base === mainModel.base;
      const hasControlImage = ca.image || (ca.processedImage && ca.processorConfig);
      return hasModel && modelMatchesBase && hasControlImage;
    });
  addGlobalControlNetsToGraph(globalControlNetConfigs, g, denoise);

  const globalT2IAdapterConfigs = state.controlLayers.present.layers
    // Must be a CA layer
    .filter(isControlAdapterLayer)
    // Must be enabled
    .filter((l) => l.isEnabled)
    // We want the CAs themselves
    .map((l) => l.controlAdapter)
    // Must have a ControlNet CA
    .filter(isT2IAdapterConfigV2)
    .filter((ca) => {
      const hasModel = Boolean(ca.model);
      const modelMatchesBase = ca.model?.base === mainModel.base;
      const hasControlImage = ca.image || (ca.processedImage && ca.processorConfig);
      return hasModel && modelMatchesBase && hasControlImage;
    });
  addGlobalT2IAdaptersToGraph(globalT2IAdapterConfigs, g, denoise);

  const globalIPAdapterConfigs = state.controlLayers.present.layers
    // Must be an IP Adapter layer
    .filter(isIPAdapterLayer)
    // Must be enabled
    .filter((l) => l.isEnabled)
    // We want the IP Adapters themselves
    .map((l) => l.ipAdapter)
    .filter((ca) => {
      const hasModel = Boolean(ca.model);
      const modelMatchesBase = ca.model?.base === mainModel.base;
      const hasControlImage = Boolean(ca.image);
      return hasModel && modelMatchesBase && hasControlImage;
    });
  addGlobalIPAdaptersToGraph(globalIPAdapterConfigs, g, denoise);

  const rgLayers = state.controlLayers.present.layers
    // Only RG layers are get masks
    .filter(isRegionalGuidanceLayer)
    // Only visible layers are rendered on the canvas
    .filter((l) => l.isEnabled)
    // Only layers with prompts get added to the graph
    .filter((l) => {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasIPAdapter = l.ipAdapters.length !== 0;
      return hasTextPrompt || hasIPAdapter;
    });

  const layerIds = rgLayers.map((l) => l.id);
  const blobs = await getRegionalPromptLayerBlobs(layerIds);
  assert(size(blobs) === size(layerIds), 'Mismatch between layer IDs and blobs');

  for (const layer of rgLayers) {
    const blob = blobs[layer.id];
    assert(blob, `Blob for layer ${layer.id} not found`);
    // Upload the mask image, or get the cached image if it exists
    const { image_name } = await getMaskImage(layer, blob);

    // The main mask-to-tensor node
    const maskToTensor = g.addNode({
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${layer.id}`,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    });

    if (layer.positivePrompt) {
      // The main positive conditioning node
      const regionalPosCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${layer.id}`,
              prompt: layer.positivePrompt,
              style: layer.positivePrompt, // TODO: Should we put the positive prompt in both fields?
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${layer.id}`,
              prompt: layer.positivePrompt,
            }
      );
      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', regionalPosCond, 'mask');
      // Connect the conditioning to the collector
      g.addEdge(regionalPosCond, 'conditioning', posCondCollect, 'item');
      // Copy the connections to the "global" positive conditioning node to the regional cond
      for (const edge of g.getEdgesTo(posCond)) {
        console.log(edge);
        if (edge.destination.field !== 'prompt') {
          // Clone the edge, but change the destination node to the regional conditioning node
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCond.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    if (layer.negativePrompt) {
      // The main negative conditioning node
      const regionalNegCond = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${layer.id}`,
              prompt: layer.negativePrompt,
              style: layer.negativePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${layer.id}`,
              prompt: layer.negativePrompt,
            }
      );
      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', regionalNegCond, 'mask');
      // Connect the conditioning to the collector
      g.addEdge(regionalNegCond, 'conditioning', negCondCollect, 'item');
      // Copy the connections to the "global" negative conditioning node to the regional cond
      for (const edge of g.getEdgesTo(negCond)) {
        if (edge.destination.field !== 'prompt') {
          // Clone the edge, but change the destination node to the regional conditioning node
          const clone = deepClone(edge);
          clone.destination.node_id = regionalNegCond.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    // If we are using the "invert" auto-negative setting, we need to add an additional negative conditioning node
    if (layer.autoNegative === 'invert' && layer.positivePrompt) {
      // We re-use the mask image, but invert it when converting to tensor
      const invertTensorMask = g.addNode({
        id: `${PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX}_${layer.id}`,
        type: 'invert_tensor_mask',
      });
      // Connect the OG mask image to the inverted mask-to-tensor node
      g.addEdge(maskToTensor, 'mask', invertTensorMask, 'mask');
      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the positive prompt
      const regionalPosCondInverted = g.addNode(
        isSDXL
          ? {
              type: 'sdxl_compel_prompt',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${layer.id}`,
              prompt: layer.positivePrompt,
              style: layer.positivePrompt,
            }
          : {
              type: 'compel',
              id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${layer.id}`,
              prompt: layer.positivePrompt,
            }
      );
      // Connect the inverted mask to the conditioning
      g.addEdge(invertTensorMask, 'mask', regionalPosCondInverted, 'mask');
      // Connect the conditioning to the negative collector
      g.addEdge(regionalPosCondInverted, 'conditioning', negCondCollect, 'item');
      // Copy the connections to the "global" positive conditioning node to our regional node
      for (const edge of g.getEdgesTo(posCond)) {
        if (edge.destination.field !== 'prompt') {
          // Clone the edge, but change the destination node to the regional conditioning node
          const clone = deepClone(edge);
          clone.destination.node_id = regionalPosCondInverted.id;
          g.addEdgeFromObj(clone);
        }
      }
    }

    // TODO(psyche): For some reason, I have to explicitly annotate regionalIPAdapters here. Not sure why.
    const regionalIPAdapters: IPAdapterConfigV2[] = layer.ipAdapters.filter((ipAdapter) => {
      const hasModel = Boolean(ipAdapter.model);
      const modelMatchesBase = ipAdapter.model?.base === mainModel.base;
      const hasControlImage = Boolean(ipAdapter.image);
      return hasModel && modelMatchesBase && hasControlImage;
    });

    for (const ipAdapterConfig of regionalIPAdapters) {
      const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);
      const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapterConfig;
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
          image_name: image.imageName,
        },
      });

      // Connect the mask to the conditioning
      g.addEdge(maskToTensor, 'mask', ipAdapter, 'mask');
      g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollect, 'item');
    }
  }
};

const getMaskImage = async (layer: RegionalGuidanceLayer, blob: Blob): Promise<ImageDTO> => {
  if (layer.uploadedMaskImage) {
    const imageDTO = await getImageDTO(layer.uploadedMaskImage.imageName);
    if (imageDTO) {
      return imageDTO;
    }
  }
  const { dispatch } = getStore();
  // No cached mask, or the cached image no longer exists - we need to upload the mask image
  const file = new File([blob], `${layer.id}_mask.png`, { type: 'image/png' });
  const req = dispatch(
    imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
  );
  req.reset();

  const imageDTO = await req.unwrap();
  dispatch(rgLayerMaskImageUploaded({ layerId: layer.id, imageDTO }));
  return imageDTO;
};
