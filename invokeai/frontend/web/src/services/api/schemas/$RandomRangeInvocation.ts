/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $RandomRangeInvocation = {
  description: `Creates a collection of random numbers`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    low: {
      type: 'number',
      description: `The inclusive low value`,
    },
    high: {
      type: 'number',
      description: `The exclusive high value`,
    },
    size: {
      type: 'number',
      description: `The number of values to generate`,
    },
    seed: {
      type: 'number',
      description: `The seed for the RNG (omit for random)`,
      maximum: 2147483647,
    },
  },
} as const;
