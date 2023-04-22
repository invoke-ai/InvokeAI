/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CropImageInvocation = {
  description: `Crops an image to a specified box. The box can be outside of the image.`,
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
      description: `The image to crop`,
      contains: [{
        type: 'ImageField',
      }],
    },
    'x': {
      type: 'number',
      description: `The left x coordinate of the crop rectangle`,
    },
    'y': {
      type: 'number',
      description: `The top y coordinate of the crop rectangle`,
    },
    width: {
      type: 'number',
      description: `The width of the crop rectangle`,
    },
    height: {
      type: 'number',
      description: `The height of the crop rectangle`,
    },
  },
} as const;
