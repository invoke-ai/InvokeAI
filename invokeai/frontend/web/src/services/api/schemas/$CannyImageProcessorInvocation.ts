/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CannyImageProcessorInvocation = {
  description: `Canny edge detection for ControlNet`,
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
    low_threshold: {
      type: 'number',
      description: `low threshold of Canny pixel gradient`,
    },
    high_threshold: {
      type: 'number',
      description: `high threshold of Canny pixel gradient`,
    },
  },
} as const;
