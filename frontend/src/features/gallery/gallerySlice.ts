import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import { v4 as uuidv4 } from 'uuid';

// TODO: Revise pending metadata RFC: https://github.com/lstein/stable-diffusion/issues/266
export interface SDMetadata {
  prompt?: string;
  steps?: number;
  cfgScale?: number;
  height?: number;
  width?: number;
  sampler?: string;
  seed?: number;
  img2imgStrength?: number;
  gfpganStrength?: number;
  upscalingLevel?: number;
  upscalingStrength?: number;
  initialImagePath?: string;
  maskPath?: string;
  seamless?: boolean;
  shouldFitToWidthHeight?: boolean;
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
  intermediateImage?: SDImage;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  images: [],
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
      state.currentImageUuid = newImages[0]
        ? newImages[newCurrentImageIndex].uuid
        : '';
    },
    addImage: (state, action: PayloadAction<SDImage>) => {
      state.images.push(action.payload);
      state.currentImageUuid = action.payload.uuid;
      state.intermediateImage = undefined;
    },
    setIntermediateImage: (state, action: PayloadAction<SDImage>) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    setGalleryImages: (state, action: PayloadAction<Array<string>>) => {
      // TODO: Revise pending metadata RFC: https://github.com/lstein/stable-diffusion/issues/266

      const urls = action.payload;

      if (urls.length === 0) {
        // there are no images on disk, clear the gallery
        state.images = [];
        state.currentImageUuid = '';
      } else {
        // Filter image urls that are already in the rehydrated state
        const filteredUrls = action.payload.filter(
          (url) => !state.images.find((image) => image.url === url)
        );

        // filtered array is just paths, generate needed data
        const preparedImages = filteredUrls.map((url) => {
          return {
            uuid: uuidv4(),
            url,
            metadata: {},
          };
        });

        const newImages = [...state.images].concat(preparedImages);

        state.currentImageUuid = newImages[newImages.length - 1].uuid;
        state.images = newImages;
      }
    },
  },
});

export const {
  setCurrentImage,
  deleteImage,
  addImage,
  setGalleryImages,
  setIntermediateImage,
  clearIntermediateImage,
} = gallerySlice.actions;

export default gallerySlice.reducer;
