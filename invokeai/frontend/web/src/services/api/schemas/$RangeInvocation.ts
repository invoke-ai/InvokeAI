/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $RangeInvocation = {
  description: `Creates a range`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    start: {
      type: 'number',
      description: `The start of the range`,
    },
    stop: {
      type: 'number',
      description: `The stop of the range`,
    },
    step: {
      type: 'number',
      description: `The step of the range`,
    },
  },
} as const;
