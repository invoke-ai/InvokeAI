/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InvokeAIMetadata = {
  description: `An image's InvokeAI-specific metadata`,
  properties: {
    session: {
      type: 'string',
      description: `The session that generated this image`,
    },
    source_id: {
      type: 'string',
      description: `The source id of the invocation that generated this image`,
    },
    invocation: {
      description: `The prepared invocation that generated this image`,
      properties: {
      },
    },
  },
} as const;
