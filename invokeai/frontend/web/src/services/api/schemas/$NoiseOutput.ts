/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $NoiseOutput = {
  description: `Invocation noise output`,
  properties: {
    type: {
      type: 'Enum',
    },
    noise: {
      type: 'all-of',
      description: `The output noise`,
      contains: [{
        type: 'LatentsField',
      }],
    },
  },
} as const;
