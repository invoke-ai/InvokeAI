import { createAction } from '@reduxjs/toolkit';
import { ControlNetConfig } from 'features/controlNet/store/controlNetSlice';
import { ImageDTO } from 'services/api/types';

export const canvasSavedToGallery = createAction('canvas/canvasSavedToGallery');

export const canvasMaskSavedToGallery = createAction(
  'canvas/canvasMaskSavedToGallery'
);

export const canvasCopiedToClipboard = createAction(
  'canvas/canvasCopiedToClipboard'
);

export const canvasDownloadedAsImage = createAction(
  'canvas/canvasDownloadedAsImage'
);

export const canvasMerged = createAction('canvas/canvasMerged');

export const stagingAreaImageSaved = createAction<{ imageDTO: ImageDTO }>(
  'canvas/stagingAreaImageSaved'
);

export const canvasMaskToControlNet = createAction<{
  controlNet: ControlNetConfig;
}>('canvas/canvasMaskToControlNet');

export const canvasImageToControlNet = createAction<{
  controlNet: ControlNetConfig;
}>('canvas/canvasImageToControlNet');
