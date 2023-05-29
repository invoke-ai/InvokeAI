/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $PidiImageProcessorInvocation = {
  description: `Applies PIDI processing to image`,
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
    safe: {
      type: 'boolean',
      description: `whether to use safe mode`,
    },
    scribble: {
      type: 'boolean',
      description: `whether to use scribble mode`,
    },
  },
} as const;
