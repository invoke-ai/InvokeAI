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

const exampleGraphs: Record<string, Graph> = {
  iterations: {
    nodes: {
      '0': {
        id: '0',
        type: 'range',
        start: 0,
        stop: 5,
        step: 1,
      },
      '1': {
        collection: [],
        id: '1',
        index: 0,
        type: 'iterate',
      },
      '2': {
        cfg_scale: 7.5,
        height: 512,
        id: '2',
        model: '',
        progress_images: false,
        prompt: 'dog',
        sampler_name: 'k_lms',
        seamless: false,
        steps: 11,
        type: 'txt2img',
        width: 512,
      },
    },
    edges: [
      {
        source: {
          field: 'collection',
          node_id: '0',
        },
        destination: {
          field: 'collection',
          node_id: '1',
        },
      },
      {
        source: {
          field: 'item',
          node_id: '1',
        },
        destination: {
          field: 'seed',
          node_id: '2',
        },
      },
    ],
  },
};

export const buildGraph = (state: RootState): Graph => {
  const { activeTab } = state.ui;
  const activeTabName = tabMap[activeTab];
  const nodeId = uuidv4();

  // return exampleGraphs.iterations;

  return {
    nodes: {
      [nodeId]: {
        id: nodeId,
        ...mapTabToFunction(activeTabName)(state),
      },
    },
  };
};
