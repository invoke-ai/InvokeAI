import type { ImageDTO } from 'services/api/types';

export type CropperState = {
  imageToCrop: ImageDTO | undefined;
  isCropperModalOpen: boolean;
};
