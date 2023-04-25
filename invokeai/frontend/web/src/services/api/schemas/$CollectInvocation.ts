/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CollectInvocation = {
  description: `Collects values into a collection`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    item: {
      description: `The item to collect (all inputs must be of the same type)`,
      properties: {
      },
    },
    collection: {
      type: 'array',
      contains: {
        properties: {
        },
      },
    },
  },
} as const;
