/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $IterateInvocation = {
  description: `A node to process inputs and produce outputs.
  May use dependency injection in __init__ to receive providers.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    collection: {
      type: 'array',
      contains: {
        properties: {
        },
      },
    },
    index: {
      type: 'number',
      description: `The index, will be provided on executed iterators`,
    },
  },
} as const;
