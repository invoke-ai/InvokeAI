/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $VaeRepo = {
  properties: {
    repo_id: {
      type: 'string',
      description: `The repo ID to use for this VAE`,
      isRequired: true,
    },
    path: {
      type: 'string',
      description: `The path to the VAE`,
    },
    subfolder: {
      type: 'string',
      description: `The subfolder to use for this VAE`,
    },
  },
} as const;
