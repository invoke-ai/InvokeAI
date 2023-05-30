/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $OpenposeImageProcessorInvocation = {
  description: `Applies Openpose processing to image`,
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
    hand_and_face: {
      type: 'boolean',
      description: `whether to use hands and face mode`,
    },
    detect_resolution: {
      type: 'number',
      description: `pixel resolution for edge detection`,
    },
    image_resolution: {
      type: 'number',
      description: `pixel resolution for output image`,
    },
  },
} as const;
