import type { ImageDTO } from 'services/api/types';

export type DeleteImageState = {
  imagesToDelete: ImageDTO[];
  isModalOpen: boolean;
};

export type ImageUsage = {
  isUpscaleImage: boolean;
  isRasterLayerImage: boolean;
  isInpaintMaskImage: boolean;
  isRegionalGuidanceImage: boolean;
  isNodesImage: boolean;
  isControlLayerImage: boolean;
  isReferenceImage: boolean;
};
