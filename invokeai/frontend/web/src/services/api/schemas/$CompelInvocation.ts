/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CompelInvocation = {
  description: `Parse prompt using compel package to conditioning.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    prompt: {
      type: 'string',
      description: `Prompt`,
    },
    model: {
      type: 'string',
      description: `Model to use`,
    },
  },
} as const;
