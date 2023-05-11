/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageToLatentsInvocation = {
  description: `Encodes an image into latents.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    image: {
      type: 'all-of',
      description: `The image to encode`,
      contains: [{
        type: 'ImageField',
      }],
    },
    model: {
      type: 'string',
      description: `The model to use`,
    },
  },
} as const;
