import { createAction } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';
import { SelectedImage } from 'features/parameters/store/actions';

export const requestedImageDeletion = createAction<
  Image | SelectedImage | undefined
>('gallery/requestedImageDeletion');

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const sentImageToImg2Img = createAction('gallery/sentImageToImg2Img');
