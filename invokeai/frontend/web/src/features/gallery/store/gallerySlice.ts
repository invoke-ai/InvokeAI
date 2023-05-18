import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';
import { imageReceived, thumbnailReceived } from 'services/thunks/image';
import {
  receivedResultImagesPage,
  receivedUploadImagesPage,
} from '../../../services/thunks/gallery';

type GalleryImageObjectFitType = 'contain' | 'cover';

export interface GalleryState {
  selectedImage?: Image;
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
    imageSelected: (state, action: PayloadAction<Image | undefined>) => {
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
    builder.addCase(imageReceived.fulfilled, (state, action) => {
      // When we get an updated URL for an image, we need to update the selectedImage in gallery,
      // which is currently its own object (instead of a reference to an image in results/uploads)
      const { imagePath } = action.payload;
      const { imageName } = action.meta.arg;

      if (state.selectedImage?.name === imageName) {
        state.selectedImage.url = imagePath;
      }
    });

    builder.addCase(thumbnailReceived.fulfilled, (state, action) => {
      // When we get an updated URL for an image, we need to update the selectedImage in gallery,
      // which is currently its own object (instead of a reference to an image in results/uploads)
      const { thumbnailPath } = action.payload;
      const { thumbnailName } = action.meta.arg;

      if (state.selectedImage?.name === thumbnailName) {
        state.selectedImage.thumbnail = thumbnailPath;
      }
    });
    builder.addCase(receivedResultImagesPage.fulfilled, (state, action) => {
      // rehydrate selectedImage URL when results list comes in
      // solves case when outdated URL is in local storage
      const selectedImage = state.selectedImage;
      if (selectedImage) {
        const selectedImageInResults = action.payload.items.find(
          (image) => image.image_name === selectedImage.name
        );
        if (selectedImageInResults) {
          selectedImage.url = selectedImageInResults.image_url;
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
          (image) => image.image_name === selectedImage.name
        );
        if (selectedImageInResults) {
          selectedImage.url = selectedImageInResults.image_url;
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
