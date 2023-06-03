import { RootState } from 'app/store/store';
import {
  Edge,
  ImageToImageInvocation,
  InpaintInvocation,
  IterateInvocation,
  RandomRangeInvocation,
  RangeInvocation,
  TextToImageInvocation,
} from 'services/api';
import { buildImg2ImgNode } from '../nodeBuilders/buildImageToImageNode';
import { buildTxt2ImgNode } from '../nodeBuilders/buildTextToImageNode';
import { buildRangeNode } from '../nodeBuilders/buildRangeNode';
import { buildIterateNode } from '../nodeBuilders/buildIterateNode';
import { buildEdges } from '../edgeBuilders/buildEdges';
import { log } from 'app/logging/useLogger';
import { buildInpaintNode } from '../nodeBuilders/buildInpaintNode';

const moduleLog = log.child({ namespace: 'nodes' });

const buildBaseNode = (
  nodeType: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint',
  state: RootState
):
  | TextToImageInvocation
  | ImageToImageInvocation
  | InpaintInvocation
  | undefined => {
  const overrides = {
    ...state.canvas.boundingBoxDimensions,
    is_intermediate: true,
  };

  if (nodeType === 'txt2img') {
    return buildTxt2ImgNode(state, overrides);
  }

  if (nodeType === 'img2img') {
    return buildImg2ImgNode(state, overrides);
  }

  if (nodeType === 'inpaint' || nodeType === 'outpaint') {
    return buildInpaintNode(state, overrides);
  }
};

/**
 * Builds the Canvas workflow graph and image blobs.
 */
export const buildCanvasGraphComponents = async (
  state: RootState,
  generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint'
): Promise<
  | {
      rangeNode: RangeInvocation | RandomRangeInvocation;
      iterateNode: IterateInvocation;
      baseNode:
        | TextToImageInvocation
        | ImageToImageInvocation
        | InpaintInvocation;
      edges: Edge[];
    }
  | undefined
> => {
  // The base node is a txt2img, img2img or inpaint node
  const baseNode = buildBaseNode(generationMode, state);

  if (!baseNode) {
    moduleLog.error('Problem building base node');
    return;
  }

  if (baseNode.type === 'inpaint') {
    const {
      seamSize,
      seamBlur,
      seamSteps,
      seamStrength,
      tileSize,
      infillMethod,
    } = state.generation;

    const { scaledBoundingBoxDimensions, boundingBoxScaleMethod } =
      state.canvas;

    if (boundingBoxScaleMethod !== 'none') {
      baseNode.inpaint_width = scaledBoundingBoxDimensions.width;
      baseNode.inpaint_height = scaledBoundingBoxDimensions.height;
    }

    baseNode.seam_size = seamSize;
    baseNode.seam_blur = seamBlur;
    baseNode.seam_strength = seamStrength;
    baseNode.seam_steps = seamSteps;
    baseNode.infill_method = infillMethod as InpaintInvocation['infill_method'];

    if (infillMethod === 'tile') {
      baseNode.tile_size = tileSize;
    }
  }

  // We always range and iterate nodes, no matter the iteration count
  // This is required to provide the correct seeds to the backend engine
  const rangeNode = buildRangeNode(state);
  const iterateNode = buildIterateNode();

  // Build the edges for the nodes selected.
  const edges = buildEdges(baseNode, rangeNode, iterateNode);

  return {
    rangeNode,
    iterateNode,
    baseNode,
    edges,
  };
};
