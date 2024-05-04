import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
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
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  NOISE,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
  RESIZE,
  T2I_ADAPTER_COLLECT,
} from 'features/nodes/util/graph/constants';
import { upsertMetadata } from 'features/nodes/util/graph/metadata';
import { size } from 'lodash-es';
import { getImageDTO, imagesApi } from 'services/api/endpoints/images';
import type {
  BaseModelType,
  CollectInvocation,
  ControlNetInvocation,
  Edge,
  ImageDTO,
  Invocation,
  IPAdapterInvocation,
  NonNullableGraph,
  T2IAdapterInvocation,
} from 'services/api/types';
import { assert } from 'tsafe';

export const addControlLayersToGraph = async (
  state: RootState,
  graph: NonNullableGraph,
  denoiseNodeId: string
): Promise<Layer[]> => {
  const mainModel = state.generation.model;
  assert(mainModel, 'Missing main model when building graph');
  const isSDXL = mainModel.base === 'sdxl';

  // Filter out layers with incompatible base model, missing control image
  const validLayers = state.controlLayers.present.layers.filter((l) => isValidLayer(l, mainModel.base));

  const validControlAdapters = validLayers.filter(isControlAdapterLayer).map((l) => l.controlAdapter);
  for (const ca of validControlAdapters) {
    addGlobalControlAdapterToGraph(ca, graph, denoiseNodeId);
  }

  const validIPAdapters = validLayers.filter(isIPAdapterLayer).map((l) => l.ipAdapter);
  for (const ipAdapter of validIPAdapters) {
    addGlobalIPAdapterToGraph(ipAdapter, graph, denoiseNodeId);
  }

  const initialImageLayers = validLayers.filter(isInitialImageLayer);
  assert(initialImageLayers.length <= 1, 'Only one initial image layer allowed');
  if (initialImageLayers[0]) {
    addInitialImageLayerToGraph(state, graph, denoiseNodeId, initialImageLayers[0]);
  }
  // TODO: We should probably just use conditioning collectors by default, and skip all this fanagling with re-routing
  // the existing conditioning nodes.

  // With regional prompts we have multiple conditioning nodes which much be routed into collectors. Set those up
  const posCondCollectNode: CollectInvocation = {
    id: POSITIVE_CONDITIONING_COLLECT,
    type: 'collect',
  };
  graph.nodes[POSITIVE_CONDITIONING_COLLECT] = posCondCollectNode;
  const negCondCollectNode: CollectInvocation = {
    id: NEGATIVE_CONDITIONING_COLLECT,
    type: 'collect',
  };
  graph.nodes[NEGATIVE_CONDITIONING_COLLECT] = negCondCollectNode;

  // Re-route the denoise node's OG conditioning inputs to the collect nodes
  const newEdges: Edge[] = [];
  for (const edge of graph.edges) {
    if (edge.destination.node_id === denoiseNodeId && edge.destination.field === 'positive_conditioning') {
      newEdges.push({
        source: edge.source,
        destination: {
          node_id: POSITIVE_CONDITIONING_COLLECT,
          field: 'item',
        },
      });
    } else if (edge.destination.node_id === denoiseNodeId && edge.destination.field === 'negative_conditioning') {
      newEdges.push({
        source: edge.source,
        destination: {
          node_id: NEGATIVE_CONDITIONING_COLLECT,
          field: 'item',
        },
      });
    } else {
      newEdges.push(edge);
    }
  }
  graph.edges = newEdges;

  // Connect collectors to the denoise nodes - must happen _after_ rerouting else you get cycles
  graph.edges.push({
    source: {
      node_id: POSITIVE_CONDITIONING_COLLECT,
      field: 'collection',
    },
    destination: {
      node_id: denoiseNodeId,
      field: 'positive_conditioning',
    },
  });
  graph.edges.push({
    source: {
      node_id: NEGATIVE_CONDITIONING_COLLECT,
      field: 'collection',
    },
    destination: {
      node_id: denoiseNodeId,
      field: 'negative_conditioning',
    },
  });

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
    const maskToTensorNode: Invocation<'alpha_mask_to_tensor'> = {
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${layer.id}`,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    };
    graph.nodes[maskToTensorNode.id] = maskToTensorNode;

    if (layer.positivePrompt) {
      // The main positive conditioning node
      const regionalPositiveCondNode: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'> = isSDXL
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
          };
      graph.nodes[regionalPositiveCondNode.id] = regionalPositiveCondNode;

      // Connect the mask to the conditioning
      graph.edges.push({
        source: { node_id: maskToTensorNode.id, field: 'mask' },
        destination: { node_id: regionalPositiveCondNode.id, field: 'mask' },
      });

      // Connect the conditioning to the collector
      graph.edges.push({
        source: { node_id: regionalPositiveCondNode.id, field: 'conditioning' },
        destination: { node_id: posCondCollectNode.id, field: 'item' },
      });

      // Copy the connections to the "global" positive conditioning node to the regional cond
      for (const edge of graph.edges) {
        if (edge.destination.node_id === POSITIVE_CONDITIONING && edge.destination.field !== 'prompt') {
          graph.edges.push({
            source: edge.source,
            destination: { node_id: regionalPositiveCondNode.id, field: edge.destination.field },
          });
        }
      }
    }

    if (layer.negativePrompt) {
      // The main negative conditioning node
      const regionalNegativeCondNode: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'> = isSDXL
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
          };
      graph.nodes[regionalNegativeCondNode.id] = regionalNegativeCondNode;

      // Connect the mask to the conditioning
      graph.edges.push({
        source: { node_id: maskToTensorNode.id, field: 'mask' },
        destination: { node_id: regionalNegativeCondNode.id, field: 'mask' },
      });

      // Connect the conditioning to the collector
      graph.edges.push({
        source: { node_id: regionalNegativeCondNode.id, field: 'conditioning' },
        destination: { node_id: negCondCollectNode.id, field: 'item' },
      });

      // Copy the connections to the "global" negative conditioning node to the regional cond
      for (const edge of graph.edges) {
        if (edge.destination.node_id === NEGATIVE_CONDITIONING && edge.destination.field !== 'prompt') {
          graph.edges.push({
            source: edge.source,
            destination: { node_id: regionalNegativeCondNode.id, field: edge.destination.field },
          });
        }
      }
    }

    // If we are using the "invert" auto-negative setting, we need to add an additional negative conditioning node
    if (layer.autoNegative === 'invert' && layer.positivePrompt) {
      // We re-use the mask image, but invert it when converting to tensor
      const invertTensorMaskNode: Invocation<'invert_tensor_mask'> = {
        id: `${PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX}_${layer.id}`,
        type: 'invert_tensor_mask',
      };
      graph.nodes[invertTensorMaskNode.id] = invertTensorMaskNode;

      // Connect the OG mask image to the inverted mask-to-tensor node
      graph.edges.push({
        source: {
          node_id: maskToTensorNode.id,
          field: 'mask',
        },
        destination: {
          node_id: invertTensorMaskNode.id,
          field: 'mask',
        },
      });

      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the
      // positive prompt
      const regionalPositiveCondInvertedNode: Invocation<'compel'> | Invocation<'sdxl_compel_prompt'> = isSDXL
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
          };
      graph.nodes[regionalPositiveCondInvertedNode.id] = regionalPositiveCondInvertedNode;
      // Connect the inverted mask to the conditioning
      graph.edges.push({
        source: { node_id: invertTensorMaskNode.id, field: 'mask' },
        destination: { node_id: regionalPositiveCondInvertedNode.id, field: 'mask' },
      });
      // Connect the conditioning to the negative collector
      graph.edges.push({
        source: { node_id: regionalPositiveCondInvertedNode.id, field: 'conditioning' },
        destination: { node_id: negCondCollectNode.id, field: 'item' },
      });
      // Copy the connections to the "global" positive conditioning node to our regional node
      for (const edge of graph.edges) {
        if (edge.destination.node_id === POSITIVE_CONDITIONING && edge.destination.field !== 'prompt') {
          graph.edges.push({
            source: edge.source,
            destination: { node_id: regionalPositiveCondInvertedNode.id, field: edge.destination.field },
          });
        }
      }
    }

    const validRegionalIPAdapters: IPAdapterConfigV2[] = layer.ipAdapters.filter((ipa) =>
      isValidIPAdapter(ipa, mainModel.base)
    );

    for (const ipAdapter of validRegionalIPAdapters) {
      addIPAdapterCollectorSafe(graph, denoiseNodeId);
      const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;
      assert(model, 'IP Adapter model is required');
      assert(image, 'IP Adapter image is required');

      const ipAdapterNode: IPAdapterInvocation = {
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
      };

      graph.nodes[ipAdapterNode.id] = ipAdapterNode;

      // Connect the mask to the conditioning
      graph.edges.push({
        source: { node_id: maskToTensorNode.id, field: 'mask' },
        destination: { node_id: ipAdapterNode.id, field: 'mask' },
      });

      graph.edges.push({
        source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
        destination: {
          node_id: IP_ADAPTER_COLLECT,
          field: 'item',
        },
      });
    }
  }

  upsertMetadata(graph, { control_layers: { layers: validLayers, version: state.controlLayers.present._version } });
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
  controlAdapter: ControlNetConfigV2 | T2IAdapterConfigV2,
  graph: NonNullableGraph,
  denoiseNodeId: string
) => {
  if (controlAdapter.type === 'controlnet') {
    addGlobalControlNetToGraph(controlAdapter, graph, denoiseNodeId);
  }
  if (controlAdapter.type === 't2i_adapter') {
    addGlobalT2IAdapterToGraph(controlAdapter, graph, denoiseNodeId);
  }
};

const addControlNetCollectorSafe = (graph: NonNullableGraph, denoiseNodeId: string) => {
  if (graph.nodes[CONTROL_NET_COLLECT]) {
    // You see, we've already got one!
    return;
  }
  // Add the ControlNet collector
  const controlNetIterateNode: CollectInvocation = {
    id: CONTROL_NET_COLLECT,
    type: 'collect',
    is_intermediate: true,
  };
  graph.nodes[CONTROL_NET_COLLECT] = controlNetIterateNode;
  graph.edges.push({
    source: { node_id: CONTROL_NET_COLLECT, field: 'collection' },
    destination: {
      node_id: denoiseNodeId,
      field: 'control',
    },
  });
};

const addGlobalControlNetToGraph = (controlNet: ControlNetConfigV2, graph: NonNullableGraph, denoiseNodeId: string) => {
  const { id, beginEndStepPct, controlMode, image, model, processedImage, processorConfig, weight } = controlNet;
  assert(model, 'ControlNet model is required');
  const controlImage = buildControlImage(image, processedImage, processorConfig);
  addControlNetCollectorSafe(graph, denoiseNodeId);

  const controlNetNode: ControlNetInvocation = {
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
  };

  graph.nodes[controlNetNode.id] = controlNetNode;

  graph.edges.push({
    source: { node_id: controlNetNode.id, field: 'control' },
    destination: {
      node_id: CONTROL_NET_COLLECT,
      field: 'item',
    },
  });
};

const addT2IAdapterCollectorSafe = (graph: NonNullableGraph, denoiseNodeId: string) => {
  if (graph.nodes[T2I_ADAPTER_COLLECT]) {
    // You see, we've already got one!
    return;
  }
  // Even though denoise_latents' t2i adapter input is collection or scalar, keep it simple and always use a collect
  const t2iAdapterCollectNode: CollectInvocation = {
    id: T2I_ADAPTER_COLLECT,
    type: 'collect',
    is_intermediate: true,
  };
  graph.nodes[T2I_ADAPTER_COLLECT] = t2iAdapterCollectNode;
  graph.edges.push({
    source: { node_id: T2I_ADAPTER_COLLECT, field: 'collection' },
    destination: {
      node_id: denoiseNodeId,
      field: 't2i_adapter',
    },
  });
};

const addGlobalT2IAdapterToGraph = (t2iAdapter: T2IAdapterConfigV2, graph: NonNullableGraph, denoiseNodeId: string) => {
  const { id, beginEndStepPct, image, model, processedImage, processorConfig, weight } = t2iAdapter;
  assert(model, 'T2I Adapter model is required');
  const controlImage = buildControlImage(image, processedImage, processorConfig);
  addT2IAdapterCollectorSafe(graph, denoiseNodeId);

  const t2iAdapterNode: T2IAdapterInvocation = {
    id: `t2i_adapter_${id}`,
    type: 't2i_adapter',
    is_intermediate: true,
    begin_step_percent: beginEndStepPct[0],
    end_step_percent: beginEndStepPct[1],
    resize_mode: 'just_resize',
    t2i_adapter_model: model,
    weight: weight,
    image: controlImage,
  };

  graph.nodes[t2iAdapterNode.id] = t2iAdapterNode;

  graph.edges.push({
    source: { node_id: t2iAdapterNode.id, field: 't2i_adapter' },
    destination: {
      node_id: T2I_ADAPTER_COLLECT,
      field: 'item',
    },
  });
};

const addIPAdapterCollectorSafe = (graph: NonNullableGraph, denoiseNodeId: string) => {
  if (graph.nodes[IP_ADAPTER_COLLECT]) {
    // You see, we've already got one!
    return;
  }

  const ipAdapterCollectNode: CollectInvocation = {
    id: IP_ADAPTER_COLLECT,
    type: 'collect',
    is_intermediate: true,
  };
  graph.nodes[IP_ADAPTER_COLLECT] = ipAdapterCollectNode;
  graph.edges.push({
    source: { node_id: IP_ADAPTER_COLLECT, field: 'collection' },
    destination: {
      node_id: denoiseNodeId,
      field: 'ip_adapter',
    },
  });
};

const addGlobalIPAdapterToGraph = (ipAdapter: IPAdapterConfigV2, graph: NonNullableGraph, denoiseNodeId: string) => {
  addIPAdapterCollectorSafe(graph, denoiseNodeId);
  const { id, weight, model, clipVisionModel, method, beginEndStepPct, image } = ipAdapter;
  assert(image, 'IP Adapter image is required');
  assert(model, 'IP Adapter model is required');

  const ipAdapterNode: IPAdapterInvocation = {
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
  };

  graph.nodes[ipAdapterNode.id] = ipAdapterNode;

  graph.edges.push({
    source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
    destination: {
      node_id: IP_ADAPTER_COLLECT,
      field: 'item',
    },
  });
};

const addInitialImageLayerToGraph = (
  state: RootState,
  graph: NonNullableGraph,
  denoiseNodeId: string,
  layer: InitialImageLayer
) => {
  const { vaePrecision, model } = state.generation;
  const { refinerModel, refinerStart } = state.sdxl;
  const { width, height } = state.controlLayers.present.size;
  assert(layer.isEnabled, 'Initial image layer is not enabled');
  assert(layer.image, 'Initial image layer has no image');

  const isSDXL = model?.base === 'sdxl';
  const useRefinerStartEnd = isSDXL && Boolean(refinerModel);

  const denoiseNode = graph.nodes[denoiseNodeId];
  assert(denoiseNode?.type === 'denoise_latents', `Missing denoise node or incorrect type: ${denoiseNode?.type}`);

  const { denoisingStrength } = layer;
  denoiseNode.denoising_start = useRefinerStartEnd
    ? Math.min(refinerStart, 1 - denoisingStrength)
    : 1 - denoisingStrength;
  denoiseNode.denoising_end = useRefinerStartEnd ? refinerStart : 1;

  const i2lNode: Invocation<'i2l'> = {
    type: 'i2l',
    id: IMAGE_TO_LATENTS,
    is_intermediate: true,
    use_cache: true,
    fp32: vaePrecision === 'fp32',
  };

  graph.nodes[i2lNode.id] = i2lNode;
  graph.edges.push({
    source: {
      node_id: IMAGE_TO_LATENTS,
      field: 'latents',
    },
    destination: {
      node_id: denoiseNode.id,
      field: 'latents',
    },
  });

  if (layer.image.width !== width || layer.image.height !== height) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: Invocation<'img_resize'> = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: layer.image.name,
      },
      is_intermediate: true,
      width,
      height,
    };

    graph.nodes[RESIZE] = resizeNode;

    // The `RESIZE` node then passes its image to `IMAGE_TO_LATENTS`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'image' },
      destination: {
        node_id: IMAGE_TO_LATENTS,
        field: 'image',
      },
    });

    // The `RESIZE` node also passes its width and height to `NOISE`
    graph.edges.push({
      source: { node_id: RESIZE, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });

    graph.edges.push({
      source: { node_id: RESIZE, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  } else {
    // We are not resizing, so we need to set the image on the `IMAGE_TO_LATENTS` node explicitly
    i2lNode.image = {
      image_name: layer.image.name,
    };

    // Pass the image's dimensions to the `NOISE` node
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'width' },
      destination: {
        node_id: NOISE,
        field: 'width',
      },
    });
    graph.edges.push({
      source: { node_id: IMAGE_TO_LATENTS, field: 'height' },
      destination: {
        node_id: NOISE,
        field: 'height',
      },
    });
  }

  upsertMetadata(graph, { generation_mode: isSDXL ? 'sdxl_img2img' : 'img2img' });
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
