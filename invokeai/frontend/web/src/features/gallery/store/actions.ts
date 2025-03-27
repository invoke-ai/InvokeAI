import { createAction } from '@reduxjs/toolkit';
import type { ImageDTO } from 'services/api/types';

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const imageDownloaded = createAction('gallery/imageDownloaded');

export const imageCopiedToClipboard = createAction('gallery/imageCopiedToClipboard');

export const imageOpenedInNewTab = createAction('gallery/imageOpenedInNewTab');

export const imageUploadedClientSide = createAction<{
  imageDTO: ImageDTO;
  silent: boolean;
  isFirstUploadOfBatch: boolean;
}>('gallery/imageUploadedClientSide');
