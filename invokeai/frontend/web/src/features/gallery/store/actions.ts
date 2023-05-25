import { createAction } from '@reduxjs/toolkit';
import { ImageNameAndType } from 'features/parameters/store/actions';
import { ImageDTO } from 'services/api';

export const requestedImageDeletion = createAction<
  ImageDTO | ImageNameAndType | undefined
>('gallery/requestedImageDeletion');

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const sentImageToImg2Img = createAction('gallery/sentImageToImg2Img');
