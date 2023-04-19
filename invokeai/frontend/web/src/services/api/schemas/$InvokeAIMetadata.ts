/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InvokeAIMetadata = {
  properties: {
    session_id: {
      type: 'string',
      description: `The session in which this image was created`,
    },
    node: {
      type: 'all-of',
      description: `The node that created this image`,
      contains: [{
        type: 'NodeMetadata',
      }],
    },
  },
} as const;
