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
          type: 'ControlNetInvocation',
        }, {
          type: 'ImageProcessorInvocation',
        }, {
          type: 'CompelInvocation',
        }, {
          type: 'NoiseInvocation',
        }, {
          type: 'TextToLatentsInvocation',
        }, {
          type: 'LatentsToImageInvocation',
        }, {
          type: 'ResizeLatentsInvocation',
        }, {
          type: 'ScaleLatentsInvocation',
        }, {
          type: 'ImageToLatentsInvocation',
        }, {
          type: 'AddInvocation',
        }, {
          type: 'SubtractInvocation',
        }, {
          type: 'MultiplyInvocation',
        }, {
          type: 'DivideInvocation',
        }, {
          type: 'RandomIntInvocation',
        }, {
          type: 'ParamIntInvocation',
        }, {
          type: 'CvInpaintInvocation',
        }, {
          type: 'RangeInvocation',
        }, {
          type: 'RandomRangeInvocation',
        }, {
          type: 'UpscaleInvocation',
        }, {
          type: 'RestoreFaceInvocation',
        }, {
          type: 'TextToImageInvocation',
        }, {
          type: 'InfillColorInvocation',
        }, {
          type: 'InfillTileInvocation',
        }, {
          type: 'InfillPatchMatchInvocation',
        }, {
          type: 'GraphInvocation',
        }, {
          type: 'IterateInvocation',
        }, {
          type: 'CollectInvocation',
        }, {
          type: 'CannyImageProcessorInvocation',
        }, {
          type: 'HedImageprocessorInvocation',
        }, {
          type: 'LineartImageProcessorInvocation',
        }, {
          type: 'LineartAnimeImageProcessorInvocation',
        }, {
          type: 'OpenposeImageProcessorInvocation',
        }, {
          type: 'MidasDepthImageProcessorInvocation',
        }, {
          type: 'NormalbaeImageProcessorInvocation',
        }, {
          type: 'MlsdImageProcessorInvocation',
        }, {
          type: 'PidiImageProcessorInvocation',
        }, {
          type: 'ContentShuffleImageProcessorInvocation',
        }, {
          type: 'LatentsToLatentsInvocation',
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
        type: 'Edge',
      },
    },
  },
} as const;
