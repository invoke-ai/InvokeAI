/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $MultiplyInvocation = {
  description: `Multiplies two numbers`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    'a': {
      type: 'number',
      description: `The first number`,
    },
    'b': {
      type: 'number',
      description: `The second number`,
    },
  },
} as const;
