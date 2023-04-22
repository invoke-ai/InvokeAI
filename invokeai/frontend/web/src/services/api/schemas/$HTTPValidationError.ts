/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $HTTPValidationError = {
  properties: {
    detail: {
      type: 'array',
      contains: {
        type: 'ValidationError',
      },
    },
  },
} as const;
