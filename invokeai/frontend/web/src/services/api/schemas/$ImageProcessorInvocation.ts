/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageProcessorInvocation = {
  description: `Base class for invocations that preprocess images for ControlNet`,
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
      description: `image to process`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
