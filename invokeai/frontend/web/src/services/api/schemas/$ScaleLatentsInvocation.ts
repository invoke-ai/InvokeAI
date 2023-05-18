/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ScaleLatentsInvocation = {
  description: `Scales latents by a given factor.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    latents: {
      type: 'all-of',
      description: `The latents to scale`,
      contains: [{
        type: 'LatentsField',
      }],
    },
    scale_factor: {
      type: 'number',
      description: `The factor by which to scale the latents`,
      isRequired: true,
    },
    mode: {
      type: 'Enum',
    },
    antialias: {
      type: 'boolean',
      description: `Whether or not to antialias (applied in bilinear and bicubic modes only)`,
    },
  },
} as const;
