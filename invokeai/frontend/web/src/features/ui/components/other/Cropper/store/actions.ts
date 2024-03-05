import { createAction } from '@reduxjs/toolkit';

export const cropperImageToGallery = createAction<{ id: string }>('cropper/cropperImageToGallery');
