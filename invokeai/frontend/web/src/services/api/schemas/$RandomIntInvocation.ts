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
  },
} as const;
