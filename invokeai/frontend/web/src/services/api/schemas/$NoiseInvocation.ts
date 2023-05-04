/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $NoiseInvocation = {
  description: `Generates latent noise.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    seed: {
      type: 'number',
      description: `The seed to use`,
      maximum: 4294967295,
    },
    width: {
      type: 'number',
      description: `The width of the resulting noise`,
      multipleOf: 8,
    },
    height: {
      type: 'number',
      description: `The height of the resulting noise`,
      multipleOf: 8,
    },
  },
} as const;
