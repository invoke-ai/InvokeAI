import type { RootState } from 'app/store/store';
import { isInitialImageLayer } from 'features/controlLayers/store/controlLayersSlice';
import { upsertMetadata } from 'features/nodes/util/graph/metadata';
import type { ImageResizeInvocation, ImageToLatentsInvocation, NonNullableGraph } from 'services/api/types';
import { assert } from 'tsafe';

import { IMAGE_TO_LATENTS, NOISE, RESIZE } from './constants';

/**
 * Returns true if an initial image was added, false if not.
 */
export const addInitialImageToLinearGraph = (
  state: RootState,
  graph: NonNullableGraph,
  denoiseNodeId: string
): boolean => {
  // Remove Existing UNet Connections
  const { img2imgStrength, vaePrecision, model } = state.generation;
  const { refinerModel, refinerStart } = state.sdxl;
  const { width, height } = state.controlLayers.present.size;
  const initialImageLayer = state.controlLayers.present.layers.find(isInitialImageLayer);
  const initialImage = initialImageLayer?.isEnabled ? initialImageLayer?.image : null;

  if (!initialImage) {
    return false;
  }

  const isSDXL = model?.base === 'sdxl';
  const useRefinerStartEnd = isSDXL && Boolean(refinerModel);

  const denoiseNode = graph.nodes[denoiseNodeId];
  assert(denoiseNode?.type === 'denoise_latents', `Missing denoise node or incorrect type: ${denoiseNode?.type}`);

  denoiseNode.denoising_start = useRefinerStartEnd ? Math.min(refinerStart, 1 - img2imgStrength) : 1 - img2imgStrength;
  denoiseNode.denoising_end = useRefinerStartEnd ? refinerStart : 1;

  // We conditionally hook the image in depending on if a resize is needed
  const i2lNode: ImageToLatentsInvocation = {
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

  if (initialImage.width !== width || initialImage.height !== height) {
    // The init image needs to be resized to the specified width and height before being passed to `IMAGE_TO_LATENTS`

    // Create a resize node, explicitly setting its image
    const resizeNode: ImageResizeInvocation = {
      id: RESIZE,
      type: 'img_resize',
      image: {
        image_name: initialImage.imageName,
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
      image_name: initialImage.imageName,
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

  upsertMetadata(graph, {
    generation_mode: isSDXL ? 'sdxl_img2img' : 'img2img',
    strength: img2imgStrength,
    init_image: initialImage.imageName,
  });

  return true;
};
