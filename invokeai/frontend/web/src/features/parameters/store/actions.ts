import { createAction } from '@reduxjs/toolkit';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';
import type { ImageDTO } from 'services/api/types';

export const initialImageSelected = createAction<ImageDTO | undefined>('generation/initialImageSelected');

export const modelSelected = createAction<ParameterModel>('generation/modelSelected');

export const setDefaultSettings = createAction('generation/setDefaultSettings');
