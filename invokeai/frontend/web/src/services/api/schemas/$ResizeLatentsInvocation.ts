/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ResizeLatentsInvocation = {
  description: `Resizes latents to explicit width/height (in pixels). Provided dimensions are floor-divided by 8.`,
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
      description: `The latents to resize`,
      contains: [{
        type: 'LatentsField',
      }],
    },
    width: {
      type: 'number',
      description: `The width to resize to (px)`,
      isRequired: true,
      minimum: 64,
      multipleOf: 8,
    },
    height: {
      type: 'number',
      description: `The height to resize to (px)`,
      isRequired: true,
      minimum: 64,
      multipleOf: 8,
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
