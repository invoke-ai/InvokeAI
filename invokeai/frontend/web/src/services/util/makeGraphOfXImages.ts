import { Graph, TextToImageInvocation } from '../api';

/**
 * Make a graph of however many images
 */
export const makeGraphOfXImages = (numberOfImages: string) =>
  Array.from(Array(numberOfImages))
    .map(
      (_val, i): TextToImageInvocation => ({
        id: i.toString(),
        type: 'txt2img',
        prompt: 'pizza',
        steps: 50,
        seed: 123,
        sampler_name: 'ddim',
      })
    )
    .reduce(
      (acc, val: TextToImageInvocation) => {
        acc.nodes![val.id] = val;
        return acc;
      },
      { nodes: {} } as Graph
    );
