import { createAction } from '@reduxjs/toolkit';

export const controlAdapterImageProcessed = createAction<{
  id: string;
}>('controlAdapters/imageProcessed');