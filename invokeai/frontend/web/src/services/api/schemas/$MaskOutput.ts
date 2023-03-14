/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $MaskOutput = {
  description: `Base class for invocations that output a mask`,
  properties: {
    type: {
      type: 'Enum',
    },
    mask: {
      type: 'all-of',
      description: `The output mask`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
