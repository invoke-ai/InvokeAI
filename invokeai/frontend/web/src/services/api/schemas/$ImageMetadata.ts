/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageMetadata = {
  description: `An image's metadata`,
  properties: {
    timestamp: {
      type: 'number',
      description: `The creation timestamp of the image`,
      isRequired: true,
    },
    width: {
      type: 'number',
      description: `The width of the image in pixels`,
      isRequired: true,
    },
    height: {
      type: 'number',
      description: `The width of the image in pixels`,
      isRequired: true,
    },
    sd_metadata: {
      description: `The image's SD-specific metadata`,
      properties: {
      },
    },
  },
} as const;
