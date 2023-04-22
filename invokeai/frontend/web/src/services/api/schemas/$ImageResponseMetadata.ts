/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ImageResponseMetadata = {
  description: `An image's metadata. Used only in HTTP responses.`,
  properties: {
    created: {
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
      description: `The height of the image in pixels`,
      isRequired: true,
    },
    invokeai: {
      type: 'all-of',
      description: `The image's InvokeAI-specific metadata`,
      contains: [{
        type: 'InvokeAIMetadata',
      }],
    },
  },
} as const;
