/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $GraphInvocation = {
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
    graph: {
      type: 'all-of',
      description: `The graph to run`,
      contains: [{
        type: 'Graph',
      }],
    },
  },
} as const;
