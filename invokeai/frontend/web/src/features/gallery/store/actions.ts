import { createAction } from '@reduxjs/toolkit';
import type { ImageUsage } from 'features/deleteImageModal/store/types';
import type { BoardDTO } from 'services/api/types';

export type RequestedBoardImagesDeletionArg = {
  board: BoardDTO;
  imagesUsage: ImageUsage;
};

export const requestedBoardImagesDeletion = createAction<RequestedBoardImagesDeletionArg>(
  'gallery/requestedBoardImagesDeletion'
);

export const sentImageToCanvas = createAction('gallery/sentImageToCanvas');

export const sentImageToImg2Img = createAction('gallery/sentImageToImg2Img');

export const imageDownloaded = createAction('gallery/imageDownloaded');
