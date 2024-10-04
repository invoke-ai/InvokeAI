import { createAction } from '@reduxjs/toolkit';

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const imageDownloaded = createAction('gallery/imageDownloaded');
