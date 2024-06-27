import { createAction } from '@reduxjs/toolkit';

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const sentImageToImg2Img = createAction('gallery/sentImageToImg2Img');

export const imageDownloaded = createAction('gallery/imageDownloaded');
