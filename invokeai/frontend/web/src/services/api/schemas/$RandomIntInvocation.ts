/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $RandomIntInvocation = {
  description: `Outputs a single random integer.`,
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
  },
} as const;
