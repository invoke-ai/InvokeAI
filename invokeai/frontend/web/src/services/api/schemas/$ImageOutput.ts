/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageOutput = {
  description: `Base class for invocations that output an image`,
  properties: {
    type: {
      type: 'Enum',
    },
    image: {
      type: 'all-of',
      description: `The output image`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
