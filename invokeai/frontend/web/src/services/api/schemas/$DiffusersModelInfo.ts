/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $DiffusersModelInfo = {
  properties: {
    description: {
      type: 'string',
      description: `A description of the model`,
    },
    format: {
      type: 'Enum',
    },
    vae: {
      type: 'all-of',
      description: `The VAE repo to use for this model`,
      contains: [{
        type: 'VaeRepo',
      }],
    },
    repo_id: {
      type: 'string',
      description: `The repo ID to use for this model`,
    },
    path: {
      type: 'string',
      description: `The path to the model`,
    },
  },
} as const;
