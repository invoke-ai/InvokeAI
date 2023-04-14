/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageField = {
  description: `An image field used for passing image objects between invocations`,
  properties: {
    image_type: {
      type: 'all-of',
      description: `The type of the image`,
      contains: [{
        type: 'ImageType',
      }],
      isRequired: true,
    },
    image_name: {
      type: 'string',
      description: `The name of the image`,
      isRequired: true,
    },
    width: {
      type: 'number',
      description: `The width of the image in pixels`,
      isRequired: true,
    },
    height: {
      type: 'number',
      description: `The height of the image in pixels`,
      isRequired: true,
    },
    mode: {
      type: 'string',
      description: `The image mode (ie pixel format)`,
      isRequired: true,
    },
    info: {
      description: `The image file's metadata`,
      properties: {
      },
      isRequired: true,
    },
  },
} as const;
