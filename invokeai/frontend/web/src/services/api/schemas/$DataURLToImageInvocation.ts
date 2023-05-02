/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $DataURLToImageInvocation = {
  description: `Outputs an image from a base 64 data URL.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    dataURL: {
      type: 'string',
      description: `The b64 data URL`,
      isRequired: true,
    },
  },
} as const;
