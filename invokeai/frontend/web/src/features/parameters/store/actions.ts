import { createAction } from '@reduxjs/toolkit';
import { isObject } from 'lodash-es';
import { ImageDTO, ResourceOrigin } from 'services/api';

export type ImageNameAndOrigin = {
  image_name: string;
  image_origin: ResourceOrigin;
};

export const isImageDTO = (image: any): image is ImageDTO => {
  return (
    image &&
    isObject(image) &&
    'image_name' in image &&
    image?.image_name !== undefined &&
    'image_origin' in image &&
    image?.image_origin !== undefined &&
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

export const initialImageSelected = createAction<ImageDTO | string | undefined>(
  'generation/initialImageSelected'
);
