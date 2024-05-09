import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import {
  isControlAdapterLayer,
  isInitialImageLayer,
  isIPAdapterLayer,
  isRegionalGuidanceLayer,
  rgLayerMaskImageUploaded,
} from 'features/controlLayers/store/controlLayersSlice';
import type { InitialImageLayer, Layer, RegionalGuidanceLayer } from 'features/controlLayers/store/types';
import type {
  ControlNetConfigV2,
  ImageWithDims,
  IPAdapterConfigV2,
  ProcessorConfig,
  T2IAdapterConfigV2,
} from 'features/controlLayers/util/controlAdapters';
import { getRegionalPromptLayerBlobs } from 'features/controlLayers/util/getLayerBlobs';
import type { ImageField } from 'features/nodes/types/common';
import {
  CONTROL_NET_COLLECT,
  IMAGE_TO_LATENTS,
  IP_ADAPTER_COLLECT,
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
  RESIZE,
  T2I_ADAPTER_COLLECT,
} from 'features/nodes/util/graph/constants';
import type { Graph } from 'features/nodes/util/graph/Graph';
import { MetadataUtil } from 'features/nodes/util/graph/MetadataUtil';
import { size } from 'lodash-es';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type { BaseModelType, ImageDTO, Invocation } from 'services/api/types';
import { assert } from 'tsafe';

export const addControlLayersToGraph = async (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  posCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  negCond: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'>,
  posCondCollect: Invocation<'collect'>,
  negCondCollect: Invocation<'collect'>,
  noise: Invocation<'noise'>
): Promise<Layer[]> => {
  const mainModel = state.generation.model;
  assert(mainModel, 'Missing main model when building graph');
  const isSDXL = mainModel.base === 'sdxl';

  // Filter out layers with incompatible base model, missing control image
  const validLayers = state.controlLayers.present.layers.filter((l) => isValidLayer(l, mainModel.base));

  const validControlAdapters = validLayers.filter(isControlAdapterLayer).map((l) => l.controlAdapter);
  for (const ca of validControlAdapters) {
    addGlobalControlAdapterToGraph(ca, g, denoise);
  }

  const validIPAdapters = validLayers.filter(isIPAdapterLayer).map((l) => l.ipAdapter);
  for (const ipAdapter of validIPAdapters) {
    addGlobalIPAdapterToGraph(ipAdapter, g, denoise);
  }

  const initialImageLayers = validLayers.filter(isInitialImageLayer);
  assert(initialImageLayers.length <= 1, 'Only one initial image layer allowed');
  if (initialImageLayers[0]) {
    addInitialImageLayerToGraph(state, g, denoise, noise, initialImageLayers[0]);
  }
  // TODO: We should probably just use conditioning collectors by default, and skip all this fanagling with re-routing
  // the existing conditioning nodes.

  const validRGLayers = validLayers.filter(isRegionalGuidanceLayer);
  const layerIds = validRGLayers.map((l) => l.id);
  const blobs = await getRegionalPromptLayerBlobs(layerIds);
  assert(size(blobs) === size(layerIds), 'Mismatch between layer IDs and blobs');

  for (const layer of validRGLayers) {
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

    const validRegionalIPAdapters: IPAdapterConfigV2[] = layer.ipAdapters.filter((ipa) =>
      isValidIPAdapter(ipa, mainModel.base)
    );

    for (const ipAdapterConfig of validRegionalIPAdapters) {
      const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);
      const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapterConfig;
      assert(model, 'IP Adapter model is required');
      assert(image, 'IP Adapter image is required');

      const ipAdapter = g.addNode({
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        is_intermediate: true,
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

  MetadataUtil.add(g, { control_layers: { layers: validLayers, version: state.controlLayers.present._version } });
  return validLayers;
};

const getMaskImage = async (layer: RegionalGuidanceLayer, blob: Blob): Promise<ImageDTO> => {
  if (layer.uploadedMaskImage) {
    const imageDTO = await getImageDTO(layer.uploadedMaskImage.name);
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

const buildControlImage = (
  image: ImageWithDims | null,
  processedImage: ImageWithDims | null,
  processorConfig: ProcessorConfig | null
): ImageField => {
  if (processedImage && processorConfig) {
    // We've processed the image in the app - use it for the control image.
    return {
      image_name: processedImage.name,
    };
  } else if (image) {
    // No processor selected, and we have an image - the user provided a processed image, use it for the control image.
    return {
      image_name: image.name,
    };
  }
  assert(false, 'Attempted to add unprocessed control image');
};

const addGlobalControlAdapterToGraph = (
  controlAdapterConfig: ControlNetConfigV2 | T2IAdapterConfigV2,
  g: Graph,
  denoise: Invocation<'denoise_latents'>
): void => {
  if (controlAdapterConfig.type === 'controlnet') {
    addGlobalControlNetToGraph(controlAdapterConfig, g, denoise);
  }
  if (controlAdapterConfig.type === 't2i_adapter') {
    addGlobalT2IAdapterToGraph(controlAdapterConfig, g, denoise);
  }
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

const addGlobalControlNetToGraph = (
  controlNetConfig: ControlNetConfigV2,
  g: Graph,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, beginEndStepPct, controlMode, image, model, processedImage, processorConfig, weight } = controlNetConfig;
  assert(model, 'ControlNet model is required');
  const controlImage = buildControlImage(image, processedImage, processorConfig);
  const controlNetCollect = addControlNetCollectorSafe(g, denoise);

  const controlNet = g.addNode({
    id: `control_net_${id}`,
    type: 'controlnet',
    is_intermediate: true,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    control_mode: controlMode,
    resize_mode: 'just_resize',
    control_model: model,
    control_weight: weight,
    image: controlImage,
  });
  g.addEdge(controlNet, 'control', controlNetCollect, 'item');
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

const addGlobalT2IAdapterToGraph = (
  t2iAdapterConfig: T2IAdapterConfigV2,
  g: Graph,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, beginEndStepPct, image, model, processedImage, processorConfig, weight } = t2iAdapterConfig;
  assert(model, 'T2I Adapter model is required');
  const controlImage = buildControlImage(image, processedImage, processorConfig);
  const t2iAdapterCollect = addT2IAdapterCollectorSafe(g, denoise);

  const t2iAdapter = g.addNode({
    id: `t2i_adapter_${id}`,
    type: 't2i_adapter',
    is_intermediate: true,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    t2i_adapter_model: model,
    weight: weight,
    image: controlImage,
  });

  g.addEdge(t2iAdapter, 't2i_adapter', t2iAdapterCollect, 'item');
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

const addGlobalIPAdapterToGraph = (
  ipAdapterConfig: IPAdapterConfigV2,
  g: Graph,
  denoise: Invocation<'denoise_latents'>
) => {
  const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapterConfig;
  assert(image, 'IP Adapter image is required');
  assert(model, 'IP Adapter model is required');
  const ipAdapterCollect = addIPAdapterCollectorSafe(g, denoise);

  const ipAdapter = g.addNode({
    id: `ip_adapter_${id}`,
    type: 'ip_adapter',
    is_intermediate: true,
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
  g.addEdge(ipAdapter, 'ip_adapter', ipAdapterCollect, 'item');
};

const addInitialImageLayerToGraph = (
  state: RootState,
  g: Graph,
  denoise: Invocation<'denoise_latents'>,
  noise: Invocation<'noise'>,
  layer: InitialImageLayer
) => {
  const { vaePrecision, model } = state.generation;
  const { refinerModel, refinerStart } = state.sdxl;
  const { width, height } = state.controlLayers.present.size;
  assert(layer.isEnabled, 'Initial image layer is not enabled');
  assert(layer.image, 'Initial image layer has no image');

  const isSDXL = model?.base === 'sdxl';
  const useRefinerStartEnd = isSDXL && Boolean(refinerModel);

  const { denoisingStrength } = layer;
  denoise.denoising_start = useRefinerStartEnd ? Math.min(refinerStart, 1 - denoisingStrength) : 1 - denoisingStrength;
  denoise.denoising_end = useRefinerStartEnd ? refinerStart : 1;

  const i2l = g.addNode({
    type: 'i2l',
    id: IMAGE_TO_LATENTS,
    is_intermediate: true,
    use_cache: true,
    fp32: vaePrecision === 'fp32',
  });

  g.addEdge(i2l, 'latents', denoise, 'latents');

  if (layer.image.width !== width || layer.image.height !== height) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resize = g.addNode({
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: layer.image.name,
      },
      is_intermediate: true,
      width,
      height,
    });

    // The `RESIZE` node then passes its image to `IMAGE_TO_LATENTS`
    g.addEdge(resize, 'image', i2l, 'image');
    // The `RESIZE` node also passes its width and height to `NOISE`
    g.addEdge(resize, 'width', noise, 'width');
    g.addEdge(resize, 'height', noise, 'height');
  } else {
    // We are not resizing, so we need to set the image on the `IMAGE_TO_LATENTS` node explicitly
    i2l.image = {
      image_name: layer.image.name,
    };

    // Pass the image's dimensions to the `NOISE` node
    g.addEdge(i2l, 'width', noise, 'width');
    g.addEdge(i2l, 'height', noise, 'height');
  }

  MetadataUtil.add(g, { generation_mode: isSDXL ? 'sdxl_img2img' : 'img2img' });
};

const isValidControlAdapter = (ca: ControlNetConfigV2 | T2IAdapterConfigV2, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ca.model);
  const modelMatchesBase = ca.model?.base === base;
  const hasControlImage = Boolean(ca.image || (ca.processedImage && ca.processorConfig));
  return hasModel && modelMatchesBase && hasControlImage;
};

const isValidIPAdapter = (ipa: IPAdapterConfigV2, base: BaseModelType): boolean => {
  // Must be have a model that matches the current base and must have a control image
  const hasModel = Boolean(ipa.model);
  const modelMatchesBase = ipa.model?.base === base;
  const hasImage = Boolean(ipa.image);
  return hasModel && modelMatchesBase && hasImage;
};

const isValidLayer = (layer: Layer, base: BaseModelType) => {
  if (!layer.isEnabled) {
    return false;
  }
  if (isControlAdapterLayer(layer)) {
    return isValidControlAdapter(layer.controlAdapter, base);
  }
  if (isIPAdapterLayer(layer)) {
    return isValidIPAdapter(layer.ipAdapter, base);
  }
  if (isInitialImageLayer(layer)) {
    if (!layer.image) {
      return false;
    }
    return true;
  }
  if (isRegionalGuidanceLayer(layer)) {
    if (layer.maskObjects.length === 0) {
      // Layer has no mask, meaning any guidance would be applied to an empty region.
      return false;
    }
    const hasTextPrompt = Boolean(layer.positivePrompt) || Boolean(layer.negativePrompt);
    const hasIPAdapter = layer.ipAdapters.filter((ipa) => isValidIPAdapter(ipa, base)).length > 0;
    return hasTextPrompt || hasIPAdapter;
  }
  return false;
};
