import { ImageDTO } from 'services/api/types';

export type ChangeBoardModalState = {
  isModalOpen: boolean;
  imagesToChange: ImageDTO[];
};
