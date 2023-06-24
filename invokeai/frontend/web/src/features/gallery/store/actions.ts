import { createAction } from '@reduxjs/toolkit';
import { ImageUsage } from 'app/contexts/DeleteImageContext';
import { ImageDTO } from 'services/api/types';

export type RequestedImageDeletionArg = {
  image: ImageDTO;
  imageUsage: ImageUsage;
};

export const requestedImageDeletion = createAction<RequestedImageDeletionArg>(
  'gallery/requestedImageDeletion'
);

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const sentImageToImg2Img = createAction('gallery/sentImageToImg2Img');
