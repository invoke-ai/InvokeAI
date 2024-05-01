import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import {
  isControlAdapterLayer,
  isIPAdapterLayer,
  isRegionalGuidanceLayer,
} from 'features/controlLayers/store/controlLayersSlice';
import {
  type ControlNetConfig,
  type ImageWithDims,
  type IPAdapterConfig,
  isControlNetConfig,
  isT2IAdapterConfig,
  type ProcessorConfig,
  type T2IAdapterConfig,
} from 'features/controlLayers/util/controlAdapters';
import { getRegionalPromptLayerBlobs } from 'features/controlLayers/util/getLayerBlobs';
import type { ImageField } from 'features/nodes/types/common';
import {
  CONTROL_NET_COLLECT,
  IP_ADAPTER_COLLECT,
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  PROMPT_REGION_INVERT_TENSOR_MASK_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
  T2I_ADAPTER_COLLECT,
} from 'features/nodes/util/graph/constants';
import { upsertMetadata } from 'features/nodes/util/graph/metadata';
import { size } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type {
  CollectInvocation,
  ControlNetInvocation,
  CoreMetadataInvocation,
  Edge,
  IPAdapterInvocation,
  NonNullableGraph,
  S,
  T2IAdapterInvocation,
} from 'services/api/types';
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

const buildControlNetMetadata = (controlNet: ControlNetConfig): S['ControlNetMetadataField'] => {
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

const addGlobalControlNetsToGraph = async (
  controlNets: ControlNetConfig[],
  graph: NonNullableGraph,
  denoiseNodeId: string
) => {
  if (controlNets.length === 0) {
    return;
  }
  const controlNetMetadata: CoreMetadataInvocation['controlnets'] = [];
  addControlNetCollectorSafe(graph, denoiseNodeId);

  for (const controlNet of controlNets) {
    if (!controlNet.model) {
      return;
    }
    const { id, beginEndStepPct, controlMode, image, model, processedImage, processorConfig, weight } = controlNet;

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
      image: buildControlImage(image, processedImage, processorConfig),
    };

    graph.nodes[controlNetNode.id] = controlNetNode;

    controlNetMetadata.push(buildControlNetMetadata(controlNet));

    graph.edges.push({
      source: { node_id: controlNetNode.id, field: 'control' },
      destination: {
        node_id: CONTROL_NET_COLLECT,
        field: 'item',
      },
    });
  }
  upsertMetadata(graph, { controlnets: controlNetMetadata });
};

const buildT2IAdapterMetadata = (t2iAdapter: T2IAdapterConfig): S['T2IAdapterMetadataField'] => {
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

const addGlobalT2IAdaptersToGraph = async (
  t2iAdapters: T2IAdapterConfig[],
  graph: NonNullableGraph,
  denoiseNodeId: string
) => {
  if (t2iAdapters.length === 0) {
    return;
  }
  const t2iAdapterMetadata: CoreMetadataInvocation['t2iAdapters'] = [];
  addT2IAdapterCollectorSafe(graph, denoiseNodeId);

  for (const t2iAdapter of t2iAdapters) {
    if (!t2iAdapter.model) {
      return;
    }
    const { id, beginEndStepPct, image, model, processedImage, processorConfig, weight } = t2iAdapter;

    const t2iAdapterNode: T2IAdapterInvocation = {
      id: `t2i_adapter_${id}`,
      type: 't2i_adapter',
      is_intermediate: true,
      begin_step_percent: beginEndStepPct[0],
      end_step_percent: beginEndStepPct[1],
      resize_mode: 'just_resize',
      t2i_adapter_model: model,
      weight: weight,
      image: buildControlImage(image, processedImage, processorConfig),
    };

    graph.nodes[t2iAdapterNode.id] = t2iAdapterNode;

    t2iAdapterMetadata.push(buildT2IAdapterMetadata(t2iAdapter));

    graph.edges.push({
      source: { node_id: t2iAdapterNode.id, field: 't2i_adapter' },
      destination: {
        node_id: T2I_ADAPTER_COLLECT,
        field: 'item',
      },
    });
  }

  upsertMetadata(graph, { t2iAdapters: t2iAdapterMetadata });
};

const buildIPAdapterMetadata = (ipAdapter: IPAdapterConfig): S['IPAdapterMetadataField'] => {
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

const addGlobalIPAdaptersToGraph = async (
  ipAdapters: IPAdapterConfig[],
  graph: NonNullableGraph,
  denoiseNodeId: string
) => {
  if (ipAdapters.length === 0) {
    return;
  }
  const ipAdapterMetdata: CoreMetadataInvocation['ipAdapters'] = [];
  addIPAdapterCollectorSafe(graph, denoiseNodeId);

  for (const ipAdapter of ipAdapters) {
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
        image_name: image.imageName,
      },
    };

    graph.nodes[ipAdapterNode.id] = ipAdapterNode;

    ipAdapterMetdata.push(buildIPAdapterMetadata(ipAdapter));

    graph.edges.push({
      source: { node_id: ipAdapterNode.id, field: 'ip_adapter' },
      destination: {
        node_id: IP_ADAPTER_COLLECT,
        field: 'item',
      },
    });
  }

  upsertMetadata(graph, { ipAdapters: ipAdapterMetdata });
};

export const addControlLayersToGraph = async (state: RootState, graph: NonNullableGraph, denoiseNodeId: string) => {
  const { dispatch } = getStore();
  const mainModel = state.generation.model;
  assert(mainModel, 'Missing main model when building graph');
  const isSDXL = mainModel.base === 'sdxl';

  // Add global control adapters
  const globalControlNets = state.controlLayers.present.layers
    // Must be a CA layer
    .filter(isControlAdapterLayer)
    // Must be enabled
    .filter((l) => l.isEnabled)
    // We want the CAs themselves
    .map((l) => l.controlAdapter)
    // Must be a ControlNet
    .filter(isControlNetConfig)
    .filter((ca) => {
      const hasModel = Boolean(ca.model);
      const modelMatchesBase = ca.model?.base === mainModel.base;
      const hasControlImage = ca.image || (ca.processedImage && ca.processorConfig);
      return hasModel && modelMatchesBase && hasControlImage;
    });
  addGlobalControlNetsToGraph(globalControlNets, graph, denoiseNodeId);

  const globalT2IAdapters = state.controlLayers.present.layers
    // Must be a CA layer
    .filter(isControlAdapterLayer)
    // Must be enabled
    .filter((l) => l.isEnabled)
    // We want the CAs themselves
    .map((l) => l.controlAdapter)
    // Must have a ControlNet CA
    .filter(isT2IAdapterConfig)
    .filter((ca) => {
      const hasModel = Boolean(ca.model);
      const modelMatchesBase = ca.model?.base === mainModel.base;
      const hasControlImage = ca.image || (ca.processedImage && ca.processorConfig);
      return hasModel && modelMatchesBase && hasControlImage;
    });
  addGlobalT2IAdaptersToGraph(globalT2IAdapters, graph, denoiseNodeId);

  const globalIPAdapters = state.controlLayers.present.layers
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
  addGlobalIPAdaptersToGraph(globalIPAdapters, graph, denoiseNodeId);

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

  // Upload the blobs to the backend, add each to graph
  // TODO: Store the uploaded image names in redux to reuse them, so long as the layer hasn't otherwise changed. This
  // would be a great perf win - not only would we skip re-uploading the same image, but we'd be able to use the node
  // cache (currently, when we re-use the same mask data, since it is a different image, the node cache is not used).
  for (const layer of rgLayers) {
    const blob = blobs[layer.id];
    assert(blob, `Blob for layer ${layer.id} not found`);

    const file = new File([blob], `${layer.id}_mask.png`, { type: 'image/png' });
    const req = dispatch(
      imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
    );
    req.reset();

    // TODO: This will raise on network error
    const { image_name } = await req.unwrap();

    // The main mask-to-tensor node
    const maskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${layer.id}`,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    };
    graph.nodes[maskToTensorNode.id] = maskToTensorNode;

    if (layer.positivePrompt) {
      // The main positive conditioning node
      const regionalPositiveCondNode: S['SDXLCompelPromptInvocation'] | S['CompelInvocation'] = isSDXL
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
      const regionalNegativeCondNode: S['SDXLCompelPromptInvocation'] | S['CompelInvocation'] = isSDXL
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
      const invertTensorMaskNode: S['InvertTensorMaskInvocation'] = {
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
      const regionalPositiveCondInvertedNode: S['SDXLCompelPromptInvocation'] | S['CompelInvocation'] = isSDXL
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

    // TODO(psyche): For some reason, I have to explicitly annotate regionalIPAdapters here. Not sure why.
    const regionalIPAdapters: IPAdapterConfig[] = layer.ipAdapters.filter((ipAdapter) => {
      const hasModel = Boolean(ipAdapter.model);
      const modelMatchesBase = ipAdapter.model?.base === mainModel.base;
      const hasControlImage = Boolean(ipAdapter.image);
      return hasModel && modelMatchesBase && hasControlImage;
    });

    for (const ipAdapter of regionalIPAdapters) {
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
          image_name: image.imageName,
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
};
