/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $Graph = {
  properties: {
    id: {
      type: 'string',
      description: `The id of this graph`,
    },
    nodes: {
      type: 'dictionary',
      contains: {
        type: 'one-of',
        contains: [{
          type: 'LoadImageInvocation',
        }, {
          type: 'ShowImageInvocation',
        }, {
          type: 'CropImageInvocation',
        }, {
          type: 'PasteImageInvocation',
        }, {
          type: 'MaskFromAlphaInvocation',
        }, {
          type: 'BlurInvocation',
        }, {
          type: 'LerpInvocation',
        }, {
          type: 'InverseLerpInvocation',
        }, {
          type: 'CvInpaintInvocation',
        }, {
          type: 'UpscaleInvocation',
        }, {
          type: 'RestoreFaceInvocation',
        }, {
          type: 'TextToImageInvocation',
        }, {
          type: 'GraphInvocation',
        }, {
          type: 'IterateInvocation',
        }, {
          type: 'CollectInvocation',
        }, {
          type: 'ImageToImageInvocation',
        }, {
          type: 'InpaintInvocation',
        }],
      },
    },
    edges: {
      type: 'array',
      contains: {
        type: 'array',
        contains: {
          properties: {
          },
        },
      },
    },
  },
} as const;
