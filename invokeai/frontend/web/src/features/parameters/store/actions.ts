import { createAction } from '@reduxjs/toolkit';
import { isObject } from 'lodash-es';
import { ImageDTO, ImageType } from 'services/api';

export type ImageNameAndType = {
  image_name: string;
  image_type: ImageType;
};

export const isImageDTO = (image: any): image is ImageDTO => {
  return (
    image &&
    isObject(image) &&
    'image_name' in image &&
    image?.image_name !== undefined &&
    'image_type' in image &&
    image?.image_type !== undefined &&
    'image_url' in image &&
    image?.image_url !== undefined &&
    'thumbnail_url' in image &&
    image?.thumbnail_url !== undefined &&
    'image_category' in image &&
    image?.image_category !== undefined &&
    'created_at' in image &&
    image?.created_at !== undefined
  );
};

export const initialImageSelected = createAction<
  ImageDTO | ImageNameAndType | undefined
>('generation/initialImageSelected');
