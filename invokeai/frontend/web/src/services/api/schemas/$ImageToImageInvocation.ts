/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageToImageInvocation = {
  description: `Generates an image using img2img.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    prompt: {
      type: 'string',
      description: `The prompt to generate an image from`,
    },
    seed: {
      type: 'number',
      description: `The seed to use (omit for random)`,
      maximum: 2147483647,
    },
    steps: {
      type: 'number',
      description: `The number of steps to use to generate the image`,
    },
    width: {
      type: 'number',
      description: `The width of the resulting image`,
      multipleOf: 8,
    },
    height: {
      type: 'number',
      description: `The height of the resulting image`,
      multipleOf: 8,
    },
    cfg_scale: {
      type: 'number',
      description: `The Classifier-Free Guidance, higher values may result in a result closer to the prompt`,
      minimum: 1,
    },
    scheduler: {
      type: 'Enum',
    },
    model: {
      type: 'string',
      description: `The model to use (currently ignored)`,
    },
    image: {
      type: 'all-of',
      description: `The input image`,
      contains: [{
        type: 'ImageField',
      }],
    },
    strength: {
      type: 'number',
      description: `The strength of the original image`,
      maximum: 1,
    },
    fit: {
      type: 'boolean',
      description: `Whether or not the result should be fit to the aspect ratio of the input image`,
    },
  },
} as const;
