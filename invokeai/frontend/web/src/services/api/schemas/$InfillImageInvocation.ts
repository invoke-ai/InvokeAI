/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InfillImageInvocation = {
  description: `Infills transparent areas of an image`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    image: {
      type: 'all-of',
      description: `The image to infill`,
      contains: [{
        type: 'ImageField',
      }],
    },
    infill_method: {
      type: 'Enum',
    },
    inpaint_fill: {
      type: 'all-of',
      description: `The solid infill method color`,
      contains: [{
        type: 'ColorField',
      }],
    },
    tile_size: {
      type: 'number',
      description: `The tile infill method size (px)`,
      minimum: 1,
    },
    seed: {
      type: 'number',
      description: `The seed to use (-1 for a random seed)`,
      maximum: 4294967295,
      minimum: -1,
    },
  },
} as const;
