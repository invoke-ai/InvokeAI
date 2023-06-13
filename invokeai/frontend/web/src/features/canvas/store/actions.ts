import { createAction } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api';

export const canvasSavedToGallery = createAction('canvas/canvasSavedToGallery');

export const canvasCopiedToClipboard = createAction(
  'canvas/canvasCopiedToClipboard'
);

export const canvasDownloadedAsImage = createAction(
  'canvas/canvasDownloadedAsImage'
);

export const canvasMerged = createAction('canvas/canvasMerged');

export const stagingAreaImageSaved = createAction<ImageDTO>(
  'canvas/stagingAreaImageSaved'
);
