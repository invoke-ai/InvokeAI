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
    width: {
      type: 'number',
      description: `The width of the noise in pixels`,
      isRequired: true,
    },
    height: {
      type: 'number',
      description: `The height of the noise in pixels`,
      isRequired: true,
    },
  },
} as const;
