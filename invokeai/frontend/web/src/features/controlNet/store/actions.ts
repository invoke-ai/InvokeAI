import { createAction } from '@reduxjs/toolkit';

export const controlNetImageProcessed = createAction<{
  controlNetId: string;
}>('controlNet/imageProcessed');
