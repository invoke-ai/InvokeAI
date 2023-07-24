import { createAction } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';
import { ImageUsage } from './types';

export const imageDeletionConfirmed = createAction<{
  imageDTO: ImageDTO;
  imageUsage: ImageUsage;
}>('imageDeletion/imageDeletionConfirmed');
