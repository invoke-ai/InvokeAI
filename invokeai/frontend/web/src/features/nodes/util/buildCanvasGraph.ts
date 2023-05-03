import { RootState } from 'app/store/store';
import { DataURLToImageInvocation, Graph } from 'services/api';
import { buildImg2ImgNode } from './linearGraphBuilder/buildImageToImageNode';
import { buildTxt2ImgNode } from './linearGraphBuilder/buildTextToImageNode';
import { buildRangeNode } from './linearGraphBuilder/buildRangeNode';
import { buildIterateNode } from './linearGraphBuilder/buildIterateNode';
import { buildEdges } from './linearGraphBuilder/buildEdges';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { getCanvasData } from 'features/canvas/util/getCanvasData';
import { getNodeType } from './getNodeType';
import { v4 as uuidv4 } from 'uuid';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ namespace: 'buildCanvasGraph' });

/**
 * Builds the Canvas workflow graph.
 */
export const buildCanvasGraph = (state: RootState): Graph | undefined => {
  const c = getCanvasData(state);

  if (!c) {
    moduleLog.error('Unable to create canvas graph');
    return;
  }

  moduleLog.debug({ data: c }, 'Built canvas data');

  const {
    baseDataURL,
    maskDataURL,
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels,
  } = c;

  const nodeType = getNodeType(
    baseIsPartiallyTransparent,
    baseIsFullyTransparent,
    doesMaskHaveBlackPixels
  );

  moduleLog.debug(`Node type ${nodeType}`);

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
