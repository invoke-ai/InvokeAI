/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PromptOutput = {
  description: `Base class for invocations that output a prompt`,
  properties: {
    type: {
      type: 'Enum',
    },
    prompt: {
      type: 'string',
      description: `The output prompt`,
    },
  },
} as const;
