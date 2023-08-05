import { ImageDTO } from 'services/api/types';

export type DeleteImageState = {
  imagesToDelete: ImageDTO[];
  isModalOpen: boolean;
};

export type ImageUsage = {
  isInitialImage: boolean;
  isCanvasImage: boolean;
  isNodesImage: boolean;
  isControlNetImage: boolean;
};
