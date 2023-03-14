/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CvInpaintInvocation = {
  description: `Simple inpaint using opencv.`,
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
      description: `The image to inpaint`,
      contains: [{
        type: 'ImageField',
      }],
    },
    mask: {
      type: 'all-of',
      description: `The mask to use when inpainting`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
