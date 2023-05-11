/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InvokeAIMetadata = {
  properties: {
    session_id: {
      type: 'string',
    },
    node: {
      type: 'dictionary',
      contains: {
        type: 'any-of',
        contains: [{
          type: 'string',
        }, {
          type: 'number',
        }, {
          type: 'number',
        }, {
          type: 'boolean',
        }, {
          type: 'MetadataImageField',
        }, {
          type: 'MetadataLatentsField',
        }, {
          type: 'MetadataColorField',
        }],
      },
    },
  },
} as const;
