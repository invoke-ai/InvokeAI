/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ModelsList = {
  properties: {
    models: {
      type: 'dictionary',
      contains: {
        type: 'one-of',
        contains: [{
          type: 'CkptModelInfo',
        }, {
          type: 'DiffusersModelInfo',
        }],
      },
      isRequired: true,
    },
  },
} as const;
