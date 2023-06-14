import { createAction } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api';

export const initialImageSelected = createAction<ImageDTO | string | undefined>(
  'generation/initialImageSelected'
);
