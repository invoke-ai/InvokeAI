/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $UpscaleInvocation = {
  description: `Upscales an image.`,
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
      description: `The input image`,
      contains: [{
        type: 'ImageField',
      }],
    },
    strength: {
      type: 'number',
      description: `The strength`,
      maximum: 1,
    },
    level: {
      type: 'Enum',
    },
  },
} as const;
