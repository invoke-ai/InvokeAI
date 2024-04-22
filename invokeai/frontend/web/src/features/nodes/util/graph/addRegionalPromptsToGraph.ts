import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import { selectAllIPAdapters } from 'features/controlAdapters/store/controlAdaptersSlice';
import {
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
} from 'features/nodes/util/graph/constants';
import { isVectorMaskLayer } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { size, sumBy } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type { CollectInvocation, Edge, IPAdapterInvocation, NonNullableGraph, S } from 'services/api/types';
import { assert } from 'tsafe';

export const addRegionalPromptsToGraph = async (state: RootState, graph: NonNullableGraph, denoiseNodeId: string) => {
  if (!state.regionalPrompts.present.isEnabled) {
    return;
  }
  const { dispatch } = getStore();
  const isSDXL = state.generation.model?.base === 'sdxl';
  const layers = state.regionalPrompts.present.layers
    // Only support vector mask layers now
    // TODO: Image masks
    .filter(isVectorMaskLayer)
    // Only visible layers are rendered on the canvas
    .filter((l) => l.isVisible)
    // Only layers with prompts get added to the graph
    .filter((l) => {
      const hasTextPrompt = Boolean(l.positivePrompt || l.negativePrompt);
      const hasIPAdapter = l.ipAdapterIds.length !== 0;
      return hasTextPrompt || hasIPAdapter;
    });

  const layerIds = layers.map((l) => l.id);
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

  if (!graph.nodes[IP_ADAPTER_COLLECT] && sumBy(layers, (l) => l.ipAdapterIds.length) > 0) {
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
  }

  // Upload the blobs to the backend, add each to graph
  // TODO: Store the uploaded image names in redux to reuse them, so long as the layer hasn't otherwise changed. This
  // would be a great perf win - not only would we skip re-uploading the same image, but we'd be able to use the node
  // cache (currently, when we re-use the same mask data, since it is a different image, the node cache is not used).
  for (const layer of layers) {
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

    for (const ipAdapterId of layer.ipAdapterIds) {
      const ipAdapter = selectAllIPAdapters(state.controlAdapters).find((ca) => ca.id === ipAdapterId);
      console.log(ipAdapter);
      if (!ipAdapter?.model) {
        return;
      }
      const { id, weight, model, clipVisionModel, method, beginStepPct, endStepPct, controlImage } = ipAdapter;

      assert(controlImage, 'IP Adapter image is required');

      const ipAdapterNode: IPAdapterInvocation = {
        id: `ip_adapter_${id}`,
        type: 'ip_adapter',
        is_intermediate: true,
        weight: weight,
        method: method,
        ip_adapter_model: model,
        clip_vision_model: clipVisionModel,
        begin_step_percent: beginStepPct,
        end_step_percent: endStepPct,
        image: {
          image_name: controlImage,
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
