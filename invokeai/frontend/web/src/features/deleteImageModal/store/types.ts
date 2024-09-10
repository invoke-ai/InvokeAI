import type { ImageDTO } from 'services/api/types';

export type DeleteImageState = {
  imagesToDelete: ImageDTO[];
  isModalOpen: boolean;
};

export type ImageUsage = {
  isNodesImage: boolean;
  isControlAdapterImage: boolean;
  isIPAdapterImage: boolean;
  isLayerImage: boolean;
};
