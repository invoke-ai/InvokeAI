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
  },
} as const;
