/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LatentsOutput = {
  description: `Base class for invocations that output latents`,
  properties: {
    type: {
      type: 'Enum',
    },
    latents: {
      type: 'all-of',
      description: `The output latents`,
      contains: [{
        type: 'LatentsField',
      }],
    },
  },
} as const;
