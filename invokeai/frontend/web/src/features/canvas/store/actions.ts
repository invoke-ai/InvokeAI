import { createAction } from '@reduxjs/toolkit';
import type { ImageDTO } from 'services/api/types';

export const canvasSavedToGallery = createAction('canvas/canvasSavedToGallery');

export const canvasMaskSavedToGallery = createAction('canvas/canvasMaskSavedToGallery');

export const canvasCopiedToClipboard = createAction('canvas/canvasCopiedToClipboard');

export const canvasDownloadedAsImage = createAction('canvas/canvasDownloadedAsImage');

export const canvasMerged = createAction('canvas/canvasMerged');

export const stagingAreaImageSaved = createAction<{ imageDTO: ImageDTO }>('canvas/stagingAreaImageSaved');

export const canvasMaskToControlAdapter = createAction<{ id: string }>('canvas/canvasMaskToControlAdapter');

export const canvasImageToControlAdapter = createAction<{ id: string }>('canvas/canvasImageToControlAdapter');
