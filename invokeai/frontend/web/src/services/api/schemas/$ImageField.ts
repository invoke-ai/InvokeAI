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
  },
} as const;
