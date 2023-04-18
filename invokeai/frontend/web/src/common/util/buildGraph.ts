import { RootState } from 'app/store';
import { InvokeTabName, tabMap } from 'features/ui/store/tabMap';
import { find } from 'lodash';
import {
  Graph,
  ImageToImageInvocation,
  TextToImageInvocation,
} from 'services/api';
import { buildHiResNode, buildImg2ImgNode } from './nodes/image2Image';
import { buildIteration } from './nodes/iteration';
import { buildTxt2ImgNode } from './nodes/text2Image';

// function mapTabToFunction(activeTabName: InvokeTabName) {
//   switch (activeTabName) {
//     case 'txt2img':
//       return buildTxt2ImgNode;

//     case 'img2img':
//       return buildImg2ImgNode;

//     default:
//       return buildTxt2ImgNode;
//   }
// }

const buildBaseNode = (
  state: RootState
): Record<string, TextToImageInvocation | ImageToImageInvocation> => {
  if (state.generation.isImageToImageEnabled) {
    return buildImg2ImgNode(state);
  }

  return buildTxt2ImgNode(state);
};

type BuildGraphOutput = {
  graph: Graph;
  nodeIdsToSubscribe: string[];
};

export const buildGraph = (state: RootState): BuildGraphOutput => {
  const { generation, postprocessing } = state;
  const { iterations, isImageToImageEnabled } = generation;
  const { hiresFix, hiresStrength } = postprocessing;

  const baseNode = buildBaseNode(state);

  let graph: Graph = { nodes: baseNode };
  const nodeIdsToSubscribe: string[] = [];

  graph = buildIteration(graph, state);

  // TODO: Is hires fix actually just img2img w/ a larger size and fit?
  // pretty sure it does stuff in latent space, and this isn't the whole story
  if (hiresFix && !isImageToImageEnabled) {
    const { node, edge } = buildHiResNode(
      baseNode as Record<string, TextToImageInvocation>,
      hiresStrength
    );
    graph = {
      nodes: {
        ...graph.nodes,
        ...node,
      },
      edges: [...(graph.edges || []), edge],
    };
    nodeIdsToSubscribe.push(Object.keys(node)[0]);
  }

  console.log('buildGraph: ', graph);

  return { graph, nodeIdsToSubscribe };
};
