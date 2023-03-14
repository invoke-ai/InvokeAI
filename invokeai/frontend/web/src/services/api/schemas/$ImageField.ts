/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageField = {
  description: `An image field used for passing image objects between invocations`,
  properties: {
    image_type: {
      type: 'string',
      description: `The type of the image`,
    },
    image_name: {
      type: 'string',
      description: `The name of the image`,
    },
  },
} as const;
