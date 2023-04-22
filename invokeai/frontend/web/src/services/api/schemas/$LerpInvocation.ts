/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LerpInvocation = {
  description: `Linear interpolation of all pixels of an image`,
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
      description: `The minimum output value`,
      maximum: 255,
    },
    max: {
      type: 'number',
      description: `The maximum output value`,
      maximum: 255,
    },
  },
} as const;
