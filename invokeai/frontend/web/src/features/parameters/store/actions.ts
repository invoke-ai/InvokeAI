import { createAction } from '@reduxjs/toolkit';
import { isObject } from 'lodash-es';
import { ImageDTO, ResourceOrigin } from 'services/api';

export type ImageNameAndOrigin = {
  image_name: string;
  image_origin: ResourceOrigin;
};

export const initialImageSelected = createAction<ImageDTO | string | undefined>(
  'generation/initialImageSelected'
);
