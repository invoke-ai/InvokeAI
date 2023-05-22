import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from '../../../services/thunks/gallery';
import { ImageDTO } from 'services/api';

type GalleryImageObjectFitType = 'contain' | 'cover';

export interface GalleryState {
  selectedImage?: ImageDTO;
  galleryImageMinimumWidth: number;
  galleryImageObjectFit: GalleryImageObjectFitType;
  shouldAutoSwitchToNewImages: boolean;
  shouldUseSingleGalleryColumn: boolean;
  currentCategory: 'results' | 'uploads';
}

export const initialGalleryState: GalleryState = {
  galleryImageMinimumWidth: 64,
  galleryImageObjectFit: 'cover',
  shouldAutoSwitchToNewImages: true,
  shouldUseSingleGalleryColumn: false,
  currentCategory: 'results',
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageSelected: (state, action: PayloadAction<ImageDTO | undefined>) => {
      state.selectedImage = action.payload;
      // TODO: if the user selects an image, disable the auto switch?
      // state.shouldAutoSwitchToNewImages = false;
    },
    setGalleryImageMinimumWidth: (state, action: PayloadAction<number>) => {
      state.galleryImageMinimumWidth = action.payload;
    },
    setGalleryImageObjectFit: (
      state,
      action: PayloadAction<GalleryImageObjectFitType>
    ) => {
      state.galleryImageObjectFit = action.payload;
    },
    setShouldAutoSwitchToNewImages: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoSwitchToNewImages = action.payload;
    },
    setCurrentCategory: (
      state,
      action: PayloadAction<'results' | 'uploads'>
    ) => {
      state.currentCategory = action.payload;
    },
    setShouldUseSingleGalleryColumn: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldUseSingleGalleryColumn = action.payload;
    },
  },
  extraReducers(builder) {
    builder.addCase(receivedResultImagesPage.fulfilled, (state, action) => {
      // rehydrate selectedImage URL when results list comes in
      // solves case when outdated URL is in local storage
      const selectedImage = state.selectedImage;
      if (selectedImage) {
        const selectedImageInResults = action.payload.items.find(
          (image) => image.image_name === selectedImage.image_name
        );

        if (selectedImageInResults) {
          selectedImage.image_url = selectedImageInResults.image_url;
          selectedImage.thumbnail_url = selectedImageInResults.thumbnail_url;
          state.selectedImage = selectedImage;
        }
      }
    });
    builder.addCase(receivedUploadImagesPage.fulfilled, (state, action) => {
      // rehydrate selectedImage URL when results list comes in
      // solves case when outdated URL is in local storage
      const selectedImage = state.selectedImage;
      if (selectedImage) {
        const selectedImageInResults = action.payload.items.find(
          (image) => image.image_name === selectedImage.image_name
        );

        if (selectedImageInResults) {
          selectedImage.image_url = selectedImageInResults.image_url;
          selectedImage.thumbnail_url = selectedImageInResults.thumbnail_url;
          state.selectedImage = selectedImage;
        }
      }
    });
  },
});

export const {
  imageSelected,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setShouldUseSingleGalleryColumn,
  setCurrentCategory,
} = gallerySlice.actions;

export default gallerySlice.reducer;
