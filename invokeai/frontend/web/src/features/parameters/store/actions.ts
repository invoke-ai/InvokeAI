import { createAction } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';
import { ImageType } from 'services/api';

export type SelectedImage = {
  name: string;
  type: ImageType;
};

export const initialImageSelected = createAction<
  Image | SelectedImage | undefined
>('generation/initialImageSelected');
