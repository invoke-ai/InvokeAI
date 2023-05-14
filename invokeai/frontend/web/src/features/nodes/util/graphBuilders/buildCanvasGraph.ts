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
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getGenerationMode } from '../getGenerationMode';
import { log } from 'app/logging/useLogger';
import { buildInpaintNode } from '../nodeBuilders/buildInpaintNode';

const moduleLog = log.child({ namespace: 'buildCanvasGraph' });

const buildBaseNode = (
  nodeType: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint',
  state: RootState
):
  | TextToImageInvocation
  | ImageToImageInvocation
  | InpaintInvocation
  | undefined => {
  if (nodeType === 'txt2img') {
    return buildTxt2ImgNode(state, state.canvas.boundingBoxDimensions);
  }

  if (nodeType === 'img2img') {
    return buildImg2ImgNode(state, state.canvas.boundingBoxDimensions);
  }

  if (nodeType === 'inpaint' || nodeType === 'outpaint') {
    return buildInpaintNode(state, state.canvas.boundingBoxDimensions);
  }
};

/**
 * Builds the Canvas workflow graph and image blobs.
 */
export const buildCanvasGraphAndBlobs = async (
  state: RootState
): Promise<
  | {
      rangeNode: RangeInvocation | RandomRangeInvocation;
      iterateNode: IterateInvocation;
      baseNode:
        | TextToImageInvocation
        | ImageToImageInvocation
        | InpaintInvocation;
      edges: Edge[];
      baseBlob: Blob;
      maskBlob: Blob;
      generationMode: 'txt2img' | 'img2img' | 'inpaint' | 'outpaint';
    }
  | undefined
> => {
  const c = await getCanvasData(state);

  if (!c) {
    moduleLog.error('Unable to create canvas graph');
    return;
  }

  const {
    baseBlob,
    maskBlob,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  } = c;

  moduleLog.debug(
    {
      data: {
        baseIsPartiallyTransparent,
        baseIsFullyTransparent,
        doesMaskHaveBlackPixels,
      },
    },
    'Built canvas data'
  );

  const generationMode = getGenerationMode(
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels
  );

  moduleLog.debug(`Generation mode: ${generationMode}`);

  // The base node is a txt2img, img2img or inpaint node
  const baseNode = buildBaseNode(generationMode, state);

  if (!baseNode) {
    moduleLog.error('Problem building base node');
    return;
  }

  if (baseNode.type === 'inpaint') {
    const { seamSize, seamBlur, seamSteps, seamStrength, tileSize } =
      state.generation;

    // generationParameters.invert_mask = shouldPreserveMaskedArea;
    // if (boundingBoxScale !== 'none') {
    //   generationParameters.inpaint_width = scaledBoundingBoxDimensions.width;
    //   generationParameters.inpaint_height = scaledBoundingBoxDimensions.height;
    // }
    baseNode.seam_size = seamSize;
    baseNode.seam_blur = seamBlur;
    baseNode.seam_strength = seamStrength;
    baseNode.seam_steps = seamSteps;
    baseNode.tile_size = tileSize;
    // baseNode.infill_method = infillMethod;
    // baseNode.force_outpaint = false;
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
    baseBlob,
    maskBlob,
    generationMode,
  };
};
