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
  const { autoNegative } = state.regionalPrompts;
  const layers = state.regionalPrompts.layers
    .filter((l) => l.kind === 'promptRegionLayer') // We only want the prompt region layers
    .filter((l) => l.isVisible); // Only visible layers are rendered on the canvas

  const layerIds = layers.map((l) => l.id); // We only need the IDs

  const blobs = await getRegionalPromptLayerBlobs(layerIds);

  console.log('blobs', blobs, 'layerIds', layerIds);
  assert(size(blobs) === size(layerIds), 'Mismatch between layer IDs and blobs');

  // Set up the conditioning collectors
  const posCondCollectNode: CollectInvocation = {
    id: POSITIVE_CONDITIONING_COLLECT,
    type: 'collect',
  };
  const negCondCollectNode: CollectInvocation = {
    id: NEGATIVE_CONDITIONING_COLLECT,
    type: 'collect',
  };
  graph.nodes[POSITIVE_CONDITIONING_COLLECT] = posCondCollectNode;
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
  // TODO: Append regional prompts to CLIP2's prompt?
  (graph.nodes[POSITIVE_CONDITIONING] as S['SDXLCompelPromptInvocation'] | S['CompelInvocation']).prompt = '';

  // Upload the blobs to the backend, add each to graph
  for (const [layerId, blob] of Object.entries(blobs)) {
    const layer = layers.find((l) => l.id === layerId);
    assert(layer, `Layer ${layerId} not found`);

    const file = new File([blob], `${layerId}_mask.png`, { type: 'image/png' });
    const req = dispatch(
      imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
    );
    req.reset();

    // TODO: this will raise an error
    const { image_name } = await req.unwrap();

    const maskImageNode: S['ImageInvocation'] = {
      id: `${PROMPT_REGION_MASK_IMAGE_PRIMITIVE_PREFIX}_${layerId}`,
      type: 'image',
      image: {
        image_name,
      },
    };
    graph.nodes[maskImageNode.id] = maskImageNode;

    const maskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
      id: `${PROMPT_REGION_MASK_TO_TENSOR_PREFIX}_${layerId}`,
      type: 'alpha_mask_to_tensor',
    };
    graph.nodes[maskToTensorNode.id] = maskToTensorNode;

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

    // Create the conditioning nodes for each region - different handling for SDXL

    const regionalPositiveCondNode: S['SDXLCompelPromptInvocation'] = {
      type: 'sdxl_compel_prompt',
      id: `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${layerId}`,
      prompt: layer.positivePrompt,
      style: layer.positivePrompt,
    };
    const regionalNegativeCondNode: S['SDXLCompelPromptInvocation'] = {
      type: 'sdxl_compel_prompt',
      id: `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${layerId}`,
      prompt: layer.negativePrompt,
      style: layer.negativePrompt,
    };
    graph.nodes[regionalPositiveCondNode.id] = regionalPositiveCondNode;
    graph.nodes[regionalNegativeCondNode.id] = regionalNegativeCondNode;
    graph.edges.push({
      source: { node_id: maskToTensorNode.id, field: 'mask' },
      destination: { node_id: regionalPositiveCondNode.id, field: 'mask' },
    });
    graph.edges.push({
      source: { node_id: maskToTensorNode.id, field: 'mask' },
      destination: { node_id: regionalNegativeCondNode.id, field: 'mask' },
    });
    graph.edges.push({
      source: { node_id: regionalPositiveCondNode.id, field: 'conditioning' },
      destination: { node_id: posCondCollectNode.id, field: 'item' },
    });
    graph.edges.push({
      source: { node_id: regionalNegativeCondNode.id, field: 'conditioning' },
      destination: { node_id: negCondCollectNode.id, field: 'item' },
    });
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

    if (autoNegative === 'invert') {
      // Add an additional negative conditioning node with the positive prompt & inverted region mask
      const invertedMaskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
        id: `${PROMPT_REGION_MASK_TO_TENSOR_INVERTED_PREFIX}_${layerId}`,
        type: 'alpha_mask_to_tensor',
        invert: true,
      };
      graph.nodes[invertedMaskToTensorNode.id] = invertedMaskToTensorNode;
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

      const regionalPositiveCondInvertedNode: S['SDXLCompelPromptInvocation'] = {
        type: 'sdxl_compel_prompt',
        id: `${PROMPT_REGION_POSITIVE_COND_INVERTED_PREFIX}_${layerId}`,
        prompt: layer.positivePrompt,
        style: layer.positivePrompt,
      };
      graph.nodes[regionalPositiveCondInvertedNode.id] = regionalPositiveCondInvertedNode;
      graph.edges.push({
        source: { node_id: invertedMaskToTensorNode.id, field: 'mask' },
        destination: { node_id: regionalPositiveCondInvertedNode.id, field: 'mask' },
      });
      graph.edges.push({
        source: { node_id: regionalPositiveCondInvertedNode.id, field: 'conditioning' },
        destination: { node_id: negCondCollectNode.id, field: 'item' },
      });
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
