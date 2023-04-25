/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageResponse = {
  description: `The response type for images`,
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
    image_url: {
      type: 'string',
      description: `The url of the image`,
      isRequired: true,
    },
    thumbnail_url: {
      type: 'string',
      description: `The url of the image's thumbnail`,
      isRequired: true,
    },
    metadata: {
      type: 'all-of',
      description: `The image's metadata`,
      contains: [{
        type: 'ImageResponseMetadata',
      }],
      isRequired: true,
    },
  },
} as const;
