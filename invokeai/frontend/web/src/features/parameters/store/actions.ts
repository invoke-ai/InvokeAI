import { createAction } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api/types';

export const initialImageSelected = createAction<ImageDTO | string | undefined>(
  'generation/initialImageSelected'
);
