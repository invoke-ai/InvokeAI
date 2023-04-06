/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $IntOutput = {
  description: `An integer output`,
  properties: {
    type: {
      type: 'Enum',
    },
    'a': {
      type: 'number',
      description: `The output integer`,
    },
  },
} as const;
