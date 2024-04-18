import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import {
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  PROMPT_REGION_MASK_IMAGE_PRIMITIVE_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_INVERTED_PREFIX,
  PROMPT_REGION_MASK_TO_TENSOR_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
} from 'features/nodes/util/graph/constants';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { size } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type { CollectInvocation, Edge, NonNullableGraph, S } from 'services/api/types';
import { assert } from 'tsafe';

export const addRegionalPromptsToGraph = async (state: RootState, graph: NonNullableGraph, denoiseNodeId: string) => {
  const { dispatch } = getStore();
  // TODO: Handle non-SDXL
  // const isSDXL = state.generation.model?.base === 'sdxl';
  const { autoNegative } = state.regionalPrompts.present;
  const layers = state.regionalPrompts.present.layers
    .filter((l) => l.kind === 'promptRegionLayer') // We only want the prompt region layers
    .filter((l) => l.isVisible); // Only visible layers are rendered on the canvas

  const layerIds = layers.map((l) => l.id); // We only need the IDs

  const blobs = await getRegionalPromptLayerBlobs(layerIds);

  console.log('blobs', blobs, 'layerIds', layerIds);
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

  // Remove the global prompt
  // TODO: Append regional prompts to CLIP2's prompt? Dunno...
  (graph.nodes[POSITIVE_CONDITIONING] as S['SDXLCompelPromptInvocation'] | S['CompelInvocation']).prompt = '';

  // Upload the blobs to the backend, add each to graph
  // TODO: Store the uploaded image names in redux to reuse them, so long as the layer hasn't otherwise changed. This
  // would be a great perf win - not only would we skip re-uploading the same image, but we'd be able to use the node
  // cache (currently, when we re-use the same mask data, since it is a different image, the node cache is not used).
  for (const [layerId, blob] of Object.entries(blobs)) {
    const layer = layers.find((l) => l.id === layerId);
    assert(layer, `Layer ${layerId} not found`);

    const file = new File([blob], `${layerId}_mask.png`, { type: 'image/png' });
    const req = dispatch(
      imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
    );
    req.reset();

    // TODO: This will raise on network error
    const { image_name } = await req.unwrap();

    // This mask (image primitive) will be fed into at least once mask-to-tensor node - two if we use the "invert" mode
    const maskImageNode: S['ImageInvocation'] = {
      id: `${PROMPT_REGION_MASK_IMAGE_PRIMITIVE_PREFIX}_${layerId}`,
      type: 'image',
      image: {
        image_name,
      },
    };
    graph.nodes[maskImageNode.id] = maskImageNode;

    // The main mask-to-tensor node
    const maskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${layerId}`,
      type: 'alpha_mask_to_tensor',
    };
    graph.nodes[maskToTensorNode.id] = maskToTensorNode;

    // Connect the mask image to the mask-to-tensor node
    graph.edges.push({
      source: {
        node_id: maskImageNode.id,
        field: 'image',
      },
      destination: {
        node_id: maskToTensorNode.id,
        field: 'image',
      },
    });

    // The main positive conditioning node
    const regionalPositiveCondNode: S['SDXLCompelPromptInvocation'] = {
      type: 'sdxl_compel_prompt',
      id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${layerId}`,
      prompt: layer.positivePrompt,
      style: layer.positivePrompt, // TODO: Should we put the positive prompt in both fields?
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

    // The main negative conditioning node
    const regionalNegativeCondNode: S['SDXLCompelPromptInvocation'] = {
      type: 'sdxl_compel_prompt',
      id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${layerId}`,
      prompt: layer.negativePrompt,
      style: layer.negativePrompt,
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

    // Copy the connections to the "global" positive and negative conditioning nodes to our regional nodes
    for (const edge of graph.edges) {
      if (edge.destination.node_id === POSITIVE_CONDITIONING && edge.destination.field !== 'prompt') {
        graph.edges.push({
          source: edge.source,
          destination: { node_id: regionalPositiveCondNode.id, field: edge.destination.field },
        });
      }
      if (edge.destination.node_id === NEGATIVE_CONDITIONING && edge.destination.field !== 'prompt') {
        graph.edges.push({
          source: edge.source,
          destination: { node_id: regionalNegativeCondNode.id, field: edge.destination.field },
        });
      }
    }

    // If we are using the "invert" auto-negative setting, we need to add an additional negative conditioning node
    if (autoNegative === 'invert') {
      // We re-use the mask image, but invert it when converting to tensor
      // TODO: Probably faster to invert the tensor from the earlier mask rather than read the mask image and convert...
      const invertedMaskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
        id: `${PROMPT_REGION_MASK_TO_TENSOR_INVERTED_PREFIX}_${layerId}`,
        type: 'alpha_mask_to_tensor',
        invert: true,
      };
      graph.nodes[invertedMaskToTensorNode.id] = invertedMaskToTensorNode;

      // Connect the OG mask image to the inverted mask-to-tensor node
      graph.edges.push({
        source: {
          node_id: maskImageNode.id,
          field: 'image',
        },
        destination: {
          node_id: invertedMaskToTensorNode.id,
          field: 'image',
        },
      });

      // Create the conditioning node. It's going to be connected to the negative cond collector, but it uses the
      // positive prompt
      const regionalPositiveCondInvertedNode: S['SDXLCompelPromptInvocation'] = {
        type: 'sdxl_compel_prompt',
        id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${layerId}`,
        prompt: layer.positivePrompt,
        style: layer.positivePrompt,
      };
      graph.nodes[regionalPositiveCondInvertedNode.id] = regionalPositiveCondInvertedNode;
      // Connect the inverted mask to the conditioning
      graph.edges.push({
        source: { node_id: invertedMaskToTensorNode.id, field: 'mask' },
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
  }
};
