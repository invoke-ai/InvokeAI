import { v4 as uuidv4 } from 'uuid';

import { RootState } from 'app/store';
import { InvokeTabName, tabMap } from 'features/ui/store/tabMap';
import { Graph } from 'services/api';
import { buildImg2ImgNode, buildTxt2ImgNode } from './buildNodes';

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

export const buildGraph = (state: RootState): Graph => {
  const { activeTab } = state.ui;
  const activeTabName = tabMap[activeTab];
  const nodeId = uuidv4();

  return {
    nodes: {
      [nodeId]: {
        id: nodeId,
        ...mapTabToFunction(activeTabName)(state),
      },
    },
  };
};
