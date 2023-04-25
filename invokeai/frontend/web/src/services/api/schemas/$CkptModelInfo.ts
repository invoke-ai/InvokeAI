/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $CkptModelInfo = {
  properties: {
    description: {
      type: 'string',
      description: `A description of the model`,
    },
    format: {
      type: 'Enum',
    },
    config: {
      type: 'string',
      description: `The path to the model config`,
      isRequired: true,
    },
    weights: {
      type: 'string',
      description: `The path to the model weights`,
      isRequired: true,
    },
    vae: {
      type: 'string',
      description: `The path to the model VAE`,
      isRequired: true,
    },
    width: {
      type: 'number',
      description: `The width of the model`,
    },
    height: {
      type: 'number',
      description: `The height of the model`,
    },
  },
} as const;
