/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $MidasDepthImageProcessorInvocation = {
  description: `Applies Midas depth processing to image`,
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
    a_mult: {
      type: 'number',
      description: `Midas parameter a = amult * PI`,
    },
    bg_th: {
      type: 'number',
      description: `Midas parameter bg_th`,
    },
  },
} as const;
