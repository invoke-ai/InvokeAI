import { createAction } from '@reduxjs/toolkit';
import type { ParameterModel } from 'features/parameters/types/parameterSchemas';

export const modelSelected = createAction<ParameterModel>('generation/modelSelected');

export const setDefaultSettings = createAction('generation/setDefaultSettings');
