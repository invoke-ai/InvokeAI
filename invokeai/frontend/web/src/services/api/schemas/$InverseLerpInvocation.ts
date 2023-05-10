/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InverseLerpInvocation = {
  description: `Inverse linear interpolation of all pixels of an image`,
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
      description: `The image to lerp`,
      contains: [{
        type: 'ImageField',
      }],
    },
    min: {
      type: 'number',
      description: `The minimum input value`,
      maximum: 255,
    },
    max: {
      type: 'number',
      description: `The maximum input value`,
      maximum: 255,
    },
  },
} as const;
