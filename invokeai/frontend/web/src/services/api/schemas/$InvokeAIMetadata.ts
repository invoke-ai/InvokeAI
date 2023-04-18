/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InvokeAIMetadata = {
  description: `An image's InvokeAI-specific metadata`,
  properties: {
    session_id: {
      type: 'string',
      description: `The session that generated this image`,
      isRequired: true,
    },
    invocation: {
      description: `The prepared invocation that generated this image`,
      properties: {
      },
    },
  },
} as const;
