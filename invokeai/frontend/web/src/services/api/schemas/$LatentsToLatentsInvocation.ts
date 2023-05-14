/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LatentsToLatentsInvocation = {
  description: `Generates latents using latents as base image.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    positive_conditioning: {
      type: 'all-of',
      description: `Positive conditioning for generation`,
      contains: [{
        type: 'ConditioningField',
      }],
    },
    negative_conditioning: {
      type: 'all-of',
      description: `Negative conditioning for generation`,
      contains: [{
        type: 'ConditioningField',
      }],
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
    model: {
      type: 'string',
      description: `The model to use (currently ignored)`,
    },
    seamless: {
      type: 'boolean',
      description: `Whether or not to generate an image that can tile without seams`,
    },
    seamless_axes: {
      type: 'string',
      description: `The axes to tile the image on, 'x' and/or 'y'`,
    },
    latents: {
      type: 'all-of',
      description: `The latents to use as a base image`,
      contains: [{
        type: 'LatentsField',
      }],
    },
    strength: {
      type: 'number',
      description: `The strength of the latents to use`,
    },
  },
} as const;
