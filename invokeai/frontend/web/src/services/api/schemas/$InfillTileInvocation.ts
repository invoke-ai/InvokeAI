/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InfillTileInvocation = {
  description: `Infills transparent areas of an image with tiles of the image`,
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
    tile_size: {
      type: 'number',
      description: `The tile size (px)`,
      minimum: 1,
    },
    seed: {
      type: 'number',
      description: `The seed to use for tile generation (omit for random)`,
      maximum: 2147483647,
    },
  },
} as const;
