import { RootState } from 'app/store';
import { InvokeTabName, tabMap } from 'features/ui/store/tabMap';
import {
  Graph,
  ImageToImageInvocation,
  TextToImageInvocation,
} from 'services/api';
import { buildImg2ImgNode } from './nodes/image2Image';
import { buildIteration } from './nodes/iteration';
import { buildTxt2ImgNode } from './nodes/text2Image';

function mapTabToFunction(activeTabName: InvokeTabName) {
  switch (activeTabName) {
    case 'txt2img':
      return buildTxt2ImgNode;

    case 'img2img':
      return buildImg2ImgNode;

    default:
      return buildTxt2ImgNode;
  }
}

const buildBaseNode = (
  state: RootState
): Record<string, TextToImageInvocation | ImageToImageInvocation> => {
  const { activeTab } = state.ui;
  const activeTabName = tabMap[activeTab];

  return mapTabToFunction(activeTabName)(state);
};

export const buildGraph = (state: RootState): Graph => {
  const { iterations } = state.generation;
  const baseNode = buildBaseNode(state);

  if (iterations > 1) {
    return buildIteration({ baseNode, iterations });
  }

  return { nodes: baseNode };
};
