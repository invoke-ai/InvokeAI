/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LatentsToImageInvocation = {
  description: `Generates an image from latents.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    latents: {
      type: 'all-of',
      description: `The latents to generate an image from`,
      contains: [{
        type: 'LatentsField',
      }],
    },
    model: {
      type: 'string',
      description: `The model to use`,
    },
  },
} as const;
