/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InfillColorInvocation = {
  description: `Infills transparent areas of an image with a solid color`,
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
      description: `The image to infill`,
      contains: [{
        type: 'ImageField',
      }],
    },
    color: {
      type: 'all-of',
      description: `The color to use to infill`,
      contains: [{
        type: 'ColorField',
      }],
    },
  },
} as const;
