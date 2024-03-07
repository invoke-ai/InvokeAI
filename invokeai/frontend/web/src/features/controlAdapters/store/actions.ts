import { createAction } from '@reduxjs/toolkit';
import type {
  ParameterControlNetModel,
  ParameterIPAdapterModel,
  ParameterT2IAdapterModel,
} from 'features/parameters/types/parameterSchemas';

export const controlAdapterImageProcessed = createAction<{
  id: string;
}>('controlAdapters/imageProcessed');

export const controlAdapterModelChanged = createAction<{
  id: string;
  model: ParameterControlNetModel | ParameterT2IAdapterModel | ParameterIPAdapterModel;
}>('controlAdapters/controlAdapterModelChanged');
