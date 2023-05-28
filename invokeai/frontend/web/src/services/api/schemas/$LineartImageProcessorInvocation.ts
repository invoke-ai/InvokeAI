/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LineartImageProcessorInvocation = {
  description: `Applies line art processing to image`,
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
      description: `image to process`,
      contains: [{
        type: 'ImageField',
      }],
    },
    detect_resolution: {
      type: 'number',
      description: `pixel resolution for edge detection`,
    },
    image_resolution: {
      type: 'number',
      description: `pixel resolution for output image`,
    },
    coarse: {
      type: 'boolean',
      description: `whether to use coarse mode`,
    },
  },
} as const;
