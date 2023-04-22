/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CreateModelRequest = {
  properties: {
    name: {
      type: 'string',
      description: `The name of the model`,
      isRequired: true,
    },
    info: {
      type: 'one-of',
      description: `The model info`,
      contains: [{
        type: 'CkptModelInfo',
      }, {
        type: 'DiffusersModelInfo',
      }],
      isRequired: true,
    },
  },
} as const;
