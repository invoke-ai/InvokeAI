/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $DownloadImageInvocation = {
  description: `Download an image from a URL.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    image_url: {
      type: 'string',
      description: `The URL to download`,
    },
  },
} as const;
