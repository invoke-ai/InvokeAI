import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import { invocationComplete } from 'services/events/actions';
import { isImageOutput } from 'services/types/guards';
import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import { imageUploaded } from 'services/thunks/image';
import { SelectedImage } from 'features/parameters/store/generationSlice';

type GalleryImageObjectFitType = 'contain' | 'cover';

export interface GalleryState {
  /**
   * The selected image
   */
  selectedImage?: SelectedImage;
  galleryImageMinimumWidth: number;
  galleryImageObjectFit: GalleryImageObjectFitType;
  shouldAutoSwitchToNewImages: boolean;
  galleryWidth: number;
  shouldUseSingleGalleryColumn: boolean;
  currentCategory: 'results' | 'uploads';
}

const initialState: GalleryState = {
  selectedImage: undefined,
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
    imageSelected: (
      state,
      action: PayloadAction<SelectedImage | undefined>
    ) => {
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
  extraReducers(builder) {
    /**
     * Invocation Complete
     */
    builder.addCase(invocationComplete, (state, action) => {
      const { data } = action.payload;
      if (isImageOutput(data.result) && state.shouldAutoSwitchToNewImages) {
        state.selectedImage = {
          name: data.result.image.image_name,
          type: 'results',
        };
      }
    });

    /**
     * Upload Image - FULFILLED
     */
    builder.addCase(imageUploaded.fulfilled, (state, action) => {
      const { response } = action.payload;

      const uploadedImage = deserializeImageResponse(response);
      state.selectedImage = { name: uploadedImage.name, type: 'uploads' };
    });
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
