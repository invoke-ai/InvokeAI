import { getStore } from 'app/store/nanostores/store';
import type { RootState } from 'app/store/store';
import {
  NEGATIVE_CONDITIONING,
  NEGATIVE_CONDITIONING_COLLECT,
  POSITIVE_CONDITIONING,
  POSITIVE_CONDITIONING_COLLECT,
  PROMPT_REGION_MASK_PREFIX,
  PROMPT_REGION_NEGATIVE_COND_PREFIX,
  PROMPT_REGION_POSITIVE_COND_PREFIX,
} from 'features/nodes/util/graph/constants';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { size } from 'lodash-es';
import { imagesApi } from 'services/api/endpoints/images';
import type { CollectInvocation, Edge, NonNullableGraph, S } from 'services/api/types';
import { assert } from 'tsafe';

export const addRegionalPromptsToGraph = async (state: RootState, graph: NonNullableGraph, denoiseNodeId: string) => {
  const { dispatch } = getStore();
  const isSDXL = state.generation.model?.base === 'sdxl';
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
  (graph.nodes[POSITIVE_CONDITIONING] as S['SDXLCompelPromptInvocation'] | S['CompelInvocation']).prompt = '';

  // Upload the blobs to the backend, add each to graph
  for (const [layerId, blob] of Object.entries(blobs)) {
    const layer = layers.find((l) => l.id === layerId);
    assert(layer, `Layer ${layerId} not found`);

    const id = `${PROMPT_REGION_MASK_PREFIX}_${layerId}`;
    const file = new File([blob], `${id}.png`, { type: 'image/png' });
    const req = dispatch(
      imagesApi.endpoints.uploadImage.initiate({ file, image_category: 'mask', is_intermediate: true })
    );
    req.reset();

    // TODO: this will raise an error
    const { image_name } = await req.unwrap();

    const alphaMaskToTensorNode: S['AlphaMaskToTensorInvocation'] = {
      id,
      type: 'alpha_mask_to_tensor',
      image: {
        image_name,
      },
    };
    graph.nodes[id] = alphaMaskToTensorNode;

    // Create the conditioning nodes for each region - different handling for SDXL

    // TODO: negative prompt
    const regionalPositiveCondNodeId = `${PROMPT_REGION_POSITIVE_COND_PREFIX}_${layerId}`;
    const regionalNegativeCondNodeId = `${PROMPT_REGION_NEGATIVE_COND_PREFIX}_${layerId}`;

    if (isSDXL) {
      graph.nodes[regionalPositiveCondNodeId] = {
        type: 'sdxl_compel_prompt',
        id: regionalPositiveCondNodeId,
        prompt: layer.positivePrompt,
      };
      graph.nodes[regionalNegativeCondNodeId] = {
        type: 'sdxl_compel_prompt',
        id: regionalNegativeCondNodeId,
        prompt: layer.negativePrompt,
      };
    } else {
      // TODO: non sdxl
      // graph.nodes[regionalCondNodeId] = {
      //   type: 'compel',
      //   id: regionalCondNodeId,
      //   prompt: layer.prompt,
      // };
    }
    graph.edges.push({
      source: { node_id: id, field: 'mask' },
      destination: { node_id: regionalPositiveCondNodeId, field: 'mask' },
    });
    graph.edges.push({
      source: { node_id: id, field: 'mask' },
      destination: { node_id: regionalNegativeCondNodeId, field: 'mask' },
    });
    graph.edges.push({
      source: { node_id: regionalPositiveCondNodeId, field: 'conditioning' },
      destination: { node_id: posCondCollectNode.id, field: 'item' },
    });
    graph.edges.push({
      source: { node_id: regionalNegativeCondNodeId, field: 'conditioning' },
      destination: { node_id: negCondCollectNode.id, field: 'item' },
    });
    for (const edge of graph.edges) {
      if (edge.destination.node_id === POSITIVE_CONDITIONING && edge.destination.field !== 'prompt') {
        graph.edges.push({
          source: edge.source,
          destination: { node_id: regionalPositiveCondNodeId, field: edge.destination.field },
        });
      }
      if (edge.destination.node_id === NEGATIVE_CONDITIONING && edge.destination.field !== 'prompt') {
        graph.edges.push({
          source: edge.source,
          destination: { node_id: regionalNegativeCondNodeId, field: edge.destination.field },
        });
      }
    }
  }
};
