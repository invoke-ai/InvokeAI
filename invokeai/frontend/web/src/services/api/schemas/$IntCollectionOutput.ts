/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $IntCollectionOutput = {
  description: `A collection of integers`,
  properties: {
    type: {
      type: 'Enum',
    },
    collection: {
      type: 'array',
      contains: {
        type: 'number',
      },
    },
  },
} as const;
