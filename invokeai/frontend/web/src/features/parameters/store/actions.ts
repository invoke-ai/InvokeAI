import { createAction } from '@reduxjs/toolkit';
import { ImageDTO, MainModelField, OnnxModelField } from 'services/api/types';

export const initialImageSelected = createAction<ImageDTO | undefined>(
  'generation/initialImageSelected'
);

export const modelSelected = createAction<MainModelField | OnnxModelField>(
  'generation/modelSelected'
);
