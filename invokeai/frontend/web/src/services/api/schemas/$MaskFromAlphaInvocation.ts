/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $MaskFromAlphaInvocation = {
  description: `Extracts the alpha channel of an image as a mask.`,
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
      description: `The image to create the mask from`,
      contains: [{
        type: 'ImageField',
      }],
    },
    invert: {
      type: 'boolean',
      description: `Whether or not to invert the mask`,
    },
  },
} as const;
