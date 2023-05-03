/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $TextToLatentsInvocation = {
  description: `Generates latents from a prompt.`,
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
    noise: {
      type: 'all-of',
      description: `The noise to use`,
      contains: [{
        type: 'LatentsField',
      }],
    },
    steps: {
      type: 'number',
      description: `The number of steps to use to generate the image`,
    },
    cfg_scale: {
      type: 'number',
      description: `The Classifier-Free Guidance, higher values may result in a result closer to the prompt`,
    },
    scheduler: {
      type: 'Enum',
    },
    seamless: {
      type: 'boolean',
      description: `Whether or not to generate an image that can tile without seams`,
    },
    seamless_axes: {
      type: 'string',
      description: `The axes to tile the image on, 'x' and/or 'y'`,
    },
    model: {
      type: 'string',
      description: `The model to use (currently ignored)`,
    },
    progress_images: {
      type: 'boolean',
      description: `Whether or not to produce progress images during generation`,
    },
  },
} as const;
