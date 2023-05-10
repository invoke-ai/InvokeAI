/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PasteImageInvocation = {
  description: `Pastes an image into another image.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    base_image: {
      type: 'all-of',
      description: `The base image`,
      contains: [{
        type: 'ImageField',
      }],
    },
    image: {
      type: 'all-of',
      description: `The image to paste`,
      contains: [{
        type: 'ImageField',
      }],
    },
    mask: {
      type: 'all-of',
      description: `The mask to use when pasting`,
      contains: [{
        type: 'ImageField',
      }],
    },
    'x': {
      type: 'number',
      description: `The left x coordinate at which to paste the image`,
    },
    'y': {
      type: 'number',
      description: `The top y coordinate at which to paste the image`,
    },
  },
} as const;
