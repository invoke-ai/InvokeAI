/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ControlOutput = {
  description: `node output for ControlNet info`,
  properties: {
    type: {
      type: 'Enum',
    },
    control: {
      type: 'all-of',
      description: `The control info dict`,
      contains: [{
        type: 'ControlField',
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
