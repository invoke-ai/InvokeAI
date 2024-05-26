import type { ImageDTO } from 'services/api/types';

export type DeleteImageState = {
  imagesToDelete: ImageDTO[];
  isModalOpen: boolean;
};

export type ImageUsage = {
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlImage: boolean;
  isControlLayerImage: boolean;
};
