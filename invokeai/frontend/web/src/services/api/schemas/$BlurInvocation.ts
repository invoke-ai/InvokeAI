/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $BlurInvocation = {
  description: `Blurs an image`,
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
      description: `The image to blur`,
      contains: [{
        type: 'ImageField',
      }],
    },
    radius: {
      type: 'number',
      description: `The blur radius`,
    },
    blur_type: {
      type: 'Enum',
    },
  },
} as const;
