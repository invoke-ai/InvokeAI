/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ParamIntInvocation = {
  description: `An integer parameter`,
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
      description: `The integer value`,
    },
  },
} as const;
