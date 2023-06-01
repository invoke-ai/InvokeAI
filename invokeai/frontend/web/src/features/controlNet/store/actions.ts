import { createAction } from '@reduxjs/toolkit';
import { ControlNetProcessorNode } from './types';

export const controlNetImageProcessed = createAction<{
  controlNetId: string;
  processorNode: ControlNetProcessorNode;
}>('controlNet/imageProcessed');
