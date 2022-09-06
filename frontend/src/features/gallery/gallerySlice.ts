import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { testImages } from '../../app/testingData';
import { v4 as uuidv4 } from 'uuid';

// TODO: Pending metadata RFC: https://github.com/lstein/stable-diffusion/issues/266
export interface SDMetadata {
  prompt: string;
}

export interface SDImage {
  // TODO: I have installed @types/uuid but cannot figure out how to use them here.
  uuid: string;
  url: string;
  metadata: SDMetadata;
}

export interface GalleryState {
  currentImageUuid: string;
  images: Array<SDImage>;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: testImages,
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<string>) => {
      const newCurrentImage = state.images.find(
        (image) => image.uuid === action.payload
      );

      if (newCurrentImage) {
        const { uuid } = newCurrentImage;
        state.currentImageUuid = uuid;
      }
    },
    deleteImage: (state, action: PayloadAction<string>) => {
      const newImages = state.images.filter(
        (image) => image.uuid !== action.payload
      );

      const imageToDeleteIndex = state.images.findIndex(
        (image) => image.uuid === action.payload
      );

      const newCurrentImageIndex = Math.min(
        Math.max(imageToDeleteIndex, 0),
        newImages.length - 1
      );

      state.images = newImages;
      state.currentImageUuid = newImages[newCurrentImageIndex].uuid;
    },
    addImage: (state, action: PayloadAction<SDImage>) => {
      state.images.push(action.payload);
      state.currentImageUuid = action.payload.uuid;
    },
    setGalleryImages: (state, action: PayloadAction<Array<any>>) => {
      const newGalleryImages = action.payload.map((url) => {
        return {
          uuid: uuidv4(),
          url,
          metadata: {
            prompt: 'test',
          },
        };
      });
      state.images = newGalleryImages;
      state.currentImageUuid =
        newGalleryImages[newGalleryImages.length - 1].uuid;
    },
  },
});

export const { setCurrentImage, deleteImage, addImage, setGalleryImages } =
  gallerySlice.actions;

export default gallerySlice.reducer;
