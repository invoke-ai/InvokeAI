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
    width: {
      type: 'number',
      description: `The width of the latents in pixels`,
      isRequired: true,
    },
    height: {
      type: 'number',
      description: `The height of the latents in pixels`,
      isRequired: true,
    },
  },
} as const;
