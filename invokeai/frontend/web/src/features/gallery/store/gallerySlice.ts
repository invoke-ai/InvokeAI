import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { ImageDTO } from 'services/api';
import { imageUpserted } from './imagesSlice';
import { imageUrlsReceived } from 'services/thunks/image';

type GalleryImageObjectFitType = 'contain' | 'cover';

export interface GalleryState {
  selectedImage?: ImageDTO;
  galleryImageMinimumWidth: number;
  galleryImageObjectFit: GalleryImageObjectFitType;
  shouldAutoSwitchToNewImages: boolean;
  shouldUseSingleGalleryColumn: boolean;
}

export const initialGalleryState: GalleryState = {
  galleryImageMinimumWidth: 64,
  galleryImageObjectFit: 'cover',
  shouldAutoSwitchToNewImages: true,
  shouldUseSingleGalleryColumn: false,
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
    setShouldUseSingleGalleryColumn: (
      state,
      action: PayloadAction<boolean>
    ) => {
      state.shouldUseSingleGalleryColumn = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(imageUpserted, (state, action) => {
      if (
        state.shouldAutoSwitchToNewImages &&
        action.payload.image_category === 'general'
      ) {
        state.selectedImage = action.payload;
      }
    });
    builder.addCase(imageUrlsReceived.fulfilled, (state, action) => {
      const { image_name, image_origin, image_url, thumbnail_url } =
        action.payload;

      if (state.selectedImage?.image_name === image_name) {
        state.selectedImage.image_url = image_url;
        state.selectedImage.thumbnail_url = thumbnail_url;
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
} = gallerySlice.actions;

export default gallerySlice.reducer;
