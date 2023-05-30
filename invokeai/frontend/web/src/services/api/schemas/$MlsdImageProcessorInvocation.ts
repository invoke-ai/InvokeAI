/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $MlsdImageProcessorInvocation = {
  description: `Applies MLSD processing to image`,
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
    thr_v: {
      type: 'number',
      description: `MLSD parameter thr_v`,
    },
    thr_d: {
      type: 'number',
      description: `MLSD parameter thr_d`,
    },
  },
} as const;
