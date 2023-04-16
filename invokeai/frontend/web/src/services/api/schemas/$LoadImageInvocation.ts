/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LoadImageInvocation = {
  description: `Load an image and provide it as output.`,
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
      description: `The image to show`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
