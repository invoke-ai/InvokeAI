/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ResizeImageInvocation = {
  description: `Resizes an image proportionately.`,
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
      description: `The image to resize`,
      contains: [{
        type: 'ImageField',
      }],
    },
    size: {
      type: 'number',
      description: `The size of the resized image's longest side`,
    },
    resample_mode: {
      type: 'Enum',
    },
  },
} as const;
