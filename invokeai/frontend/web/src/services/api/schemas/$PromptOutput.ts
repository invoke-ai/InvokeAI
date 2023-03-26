/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PromptOutput = {
  description: `Base class for invocations that output a prompt`,
  properties: {
    type: {
      type: 'Enum',
      isRequired: true,
    },
    prompt: {
      type: 'string',
      description: `The output prompt`,
      isRequired: true,
    },
  },
} as const;
