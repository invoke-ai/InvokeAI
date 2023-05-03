import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { Image } from 'app/types/invokeai';

type GalleryImageObjectFitType = 'contain' | 'cover';

export interface GalleryState {
  /**
   * The selected image
   */
  selectedImage?: Image;
  galleryImageMinimumWidth: number;
  galleryImageObjectFit: GalleryImageObjectFitType;
  shouldAutoSwitchToNewImages: boolean;
  galleryWidth: number;
  shouldUseSingleGalleryColumn: boolean;
  currentCategory: 'results' | 'uploads';
}

const initialState: GalleryState = {
  galleryImageMinimumWidth: 64,
  galleryImageObjectFit: 'cover',
  shouldAutoSwitchToNewImages: true,
  galleryWidth: 300,
  shouldUseSingleGalleryColumn: false,
  currentCategory: 'results',
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState,
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
    setGalleryWidth: (state, action: PayloadAction<number>) => {
      state.galleryWidth = action.payload;
    },
    setShouldUseSingleGalleryColumn: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldUseSingleGalleryColumn = action.payload;
    },
  },
});

export const {
  imageSelected,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setGalleryWidth,
  setShouldUseSingleGalleryColumn,
  setCurrentCategory,
} = gallerySlice.actions;

export default gallerySlice.reducer;
