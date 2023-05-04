/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InpaintInvocation = {
  description: `Generates an image using inpaint.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    prompt: {
      type: 'string',
      description: `The prompt to generate an image from`,
    },
    seed: {
      type: 'number',
      description: `The seed to use (-1 for a random seed)`,
      maximum: 4294967295,
      minimum: -1,
    },
    steps: {
      type: 'number',
      description: `The number of steps to use to generate the image`,
    },
    width: {
      type: 'number',
      description: `The width of the resulting image`,
      multipleOf: 8,
    },
    height: {
      type: 'number',
      description: `The height of the resulting image`,
      multipleOf: 8,
    },
    cfg_scale: {
      type: 'number',
      description: `The Classifier-Free Guidance, higher values may result in a result closer to the prompt`,
      exclusiveMinimum: 1,
    },
    scheduler: {
      type: 'Enum',
    },
    seamless: {
      type: 'boolean',
      description: `Whether or not to generate an image that can tile without seams`,
    },
    model: {
      type: 'string',
      description: `The model to use (currently ignored)`,
    },
    progress_images: {
      type: 'boolean',
      description: `Whether or not to produce progress images during generation`,
    },
    image: {
      type: 'all-of',
      description: `The input image`,
      contains: [{
        type: 'ImageField',
      }],
    },
    strength: {
      type: 'number',
      description: `The strength of the original image`,
      maximum: 1,
    },
    fit: {
      type: 'boolean',
      description: `Whether or not the result should be fit to the aspect ratio of the input image`,
    },
    mask: {
      type: 'all-of',
      description: `The mask`,
      contains: [{
        type: 'ImageField',
      }],
    },
    seam_size: {
      type: 'number',
      description: `The seam inpaint size (px)`,
      minimum: 1,
    },
    seam_blur: {
      type: 'number',
      description: `The seam inpaint blur radius (px)`,
    },
    seam_strength: {
      type: 'number',
      description: `The seam inpaint strength`,
      maximum: 1,
    },
    seam_steps: {
      type: 'number',
      description: `The number of steps to use for seam inpaint`,
      minimum: 1,
    },
    tile_size: {
      type: 'number',
      description: `The tile infill method size (px)`,
      minimum: 1,
    },
    infill_method: {
      type: 'Enum',
    },
    inpaint_width: {
      type: 'number',
      description: `The width of the inpaint region (px)`,
      multipleOf: 8,
    },
    inpaint_height: {
      type: 'number',
      description: `The height of the inpaint region (px)`,
      multipleOf: 8,
    },
    inpaint_fill: {
      type: 'all-of',
      description: `The solid infill method color`,
      contains: [{
        type: 'ColorField',
      }],
    },
    inpaint_replace: {
      type: 'number',
      description: `The amount by which to replace masked areas with latent noise`,
      maximum: 1,
    },
  },
} as const;
