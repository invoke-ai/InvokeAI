export const iterationGraph = {
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
      sampler_name: 'lms',
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
};
