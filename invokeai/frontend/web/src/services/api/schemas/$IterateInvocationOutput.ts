/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $IterateInvocationOutput = {
  description: `Used to connect iteration outputs. Will be expanded to a specific output.`,
  properties: {
    type: {
      type: 'Enum',
      isRequired: true,
    },
    item: {
      description: `The item being iterated over`,
      properties: {
      },
      isRequired: true,
    },
  },
} as const;
