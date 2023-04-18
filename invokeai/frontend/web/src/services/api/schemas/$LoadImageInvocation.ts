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
    image_type: {
      type: 'all-of',
      description: `The type of the image`,
      contains: [{
        type: 'ImageType',
      }],
      isRequired: true,
    },
    image_name: {
      type: 'string',
      description: `The name of the image`,
      isRequired: true,
    },
  },
} as const;
