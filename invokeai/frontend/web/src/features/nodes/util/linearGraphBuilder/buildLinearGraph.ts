import { RootState } from 'app/store/store';
import { DataURLToImageInvocation, Graph } from 'services/api';
import { buildImg2ImgNode } from './buildImageToImageNode';
import { buildTxt2ImgNode } from './buildTextToImageNode';
import { buildRangeNode } from './buildRangeNode';
import { buildIterateNode } from './buildIterateNode';
import { buildEdges } from './buildEdges';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { getCanvasDataURLs } from 'features/canvas/util/getCanvasDataURLs';
import { log } from 'console';
import { getNodeType } from '../getNodeType';
import { v4 as uuidv4 } from 'uuid';

/**
 * Builds the Linear workflow graph.
 */
export const buildLinearGraph = (state: RootState): Graph => {
  // The base node is either a txt2img or img2img node
  const baseNode = state.generation.isImageToImageEnabled
    ? buildImg2ImgNode(state)
    : buildTxt2ImgNode(state);

  // We always range and iterate nodes, no matter the iteration count
  // This is required to provide the correct seeds to the backend engine
  const rangeNode = buildRangeNode(state);
  const iterateNode = buildIterateNode();

  // Build the edges for the nodes selected.
  const edges = buildEdges(baseNode, rangeNode, iterateNode);

  // Assemble!
  const graph = {
    nodes: {
      [rangeNode.id]: rangeNode,
      [iterateNode.id]: iterateNode,
      [baseNode.id]: baseNode,
    },
    edges,
  };

  // TODO: hires fix requires latent space upscaling; we don't have nodes for this yet

  return graph;
};

/**
 * Builds the Linear workflow graph.
 */
export const buildCanvasGraph = (state: RootState): Graph => {
  const c = getCanvasDataURLs(state);

  if (!c) {
    throw 'problm creating canvas graph';
  }

  const {
    baseDataURL,
    maskDataURL,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  } = c;

  console.log({
    baseDataURL,
    maskDataURL,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  });

  const nodeType = getNodeType(
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels
  );

  console.log(nodeType);

  // The base node is either a txt2img or img2img node
  const baseNode =
    nodeType === 'img2img'
      ? buildImg2ImgNode(state, state.canvas.boundingBoxDimensions)
      : buildTxt2ImgNode(state, state.canvas.boundingBoxDimensions);

  const dataURLNode: DataURLToImageInvocation = {
    id: uuidv4(),
    type: 'dataURL_image',
    dataURL: baseDataURL,
  };

  // We always range and iterate nodes, no matter the iteration count
  // This is required to provide the correct seeds to the backend engine
  const rangeNode = buildRangeNode(state);
  const iterateNode = buildIterateNode();

  // Build the edges for the nodes selected.
  const edges = buildEdges(baseNode, rangeNode, iterateNode);

  if (baseNode.type === 'img2img') {
    edges.push({
      source: {
        node_id: dataURLNode.id,
        field: 'image',
      },
      destination: {
        node_id: baseNode.id,
        field: 'image',
      },
    });
  }

  // Assemble!
  const graph = {
    nodes: {
      [dataURLNode.id]: dataURLNode,
      [rangeNode.id]: rangeNode,
      [iterateNode.id]: iterateNode,
      [baseNode.id]: baseNode,
    },
    edges,
  };

  // TODO: hires fix requires latent space upscaling; we don't have nodes for this yet

  return graph;
};
