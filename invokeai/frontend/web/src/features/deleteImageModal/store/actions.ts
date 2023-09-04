import { createAction } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';
import { ImageUsage } from './types';

export const imageDeletionConfirmed = createAction<{
  imageDTOs: ImageDTO[];
  imagesUsage: ImageUsage[];
}>('deleteImageModal/imageDeletionConfirmed');
