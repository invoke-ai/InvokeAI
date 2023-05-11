/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CompelOutput = {
  description: `Compel parser output`,
  properties: {
    type: {
      type: 'Enum',
    },
    conditioning: {
      type: 'all-of',
      description: `Conditioning`,
      contains: [{
        type: 'ConditioningField',
      }],
    },
  },
} as const;
