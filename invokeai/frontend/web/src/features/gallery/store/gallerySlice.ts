import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import * as InvokeAI from 'app/invokeai';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { IRect } from 'konva/lib/types';
import { clamp } from 'lodash';

export type GalleryCategory = 'user' | 'result';

export type AddImagesPayload = {
  images: Array<InvokeAI.Image>;
  areMoreImagesAvailable: boolean;
  category: GalleryCategory;
};

type GalleryImageObjectFitType = 'contain' | 'cover';

export type Gallery = {
  images: InvokeAI.Image[];
  latest_mtime?: number;
  earliest_mtime?: number;
  areMoreImagesAvailable: boolean;
};

export interface GalleryState {
  currentImage?: InvokeAI.Image;
  currentImageUuid: string;
  intermediateImage?: InvokeAI.Image & {
    boundingBox?: IRect;
    generationMode?: InvokeTabName;
  };
  galleryImageMinimumWidth: number;
  galleryImageObjectFit: GalleryImageObjectFitType;
  shouldAutoSwitchToNewImages: boolean;
  categories: {
    user: Gallery;
    result: Gallery;
  };
  currentCategory: GalleryCategory;
  galleryWidth: number;
  shouldUseSingleGalleryColumn: boolean;
}

const initialState: GalleryState = {
  currentImageUuid: '',
  galleryImageMinimumWidth: 64,
  galleryImageObjectFit: 'cover',
  shouldAutoSwitchToNewImages: true,
  currentCategory: 'result',
  categories: {
    user: {
      images: [],
      latest_mtime: undefined,
      earliest_mtime: undefined,
      areMoreImagesAvailable: true,
    },
    result: {
      images: [],
      latest_mtime: undefined,
      earliest_mtime: undefined,
      areMoreImagesAvailable: true,
    },
  },
  galleryWidth: 300,
  shouldUseSingleGalleryColumn: false,
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState,
  reducers: {
    setCurrentImage: (state, action: PayloadAction<InvokeAI.Image>) => {
      state.currentImage = action.payload;
      state.currentImageUuid = action.payload.uuid;
    },
    removeImage: (
      state,
      action: PayloadAction<InvokeAI.ImageDeletedResponse>
    ) => {
      const { uuid, category } = action.payload;

      const tempImages = state.categories[category as GalleryCategory].images;

      const newImages = tempImages.filter((image) => image.uuid !== uuid);

      if (uuid === state.currentImageUuid) {
        /**
         * We are deleting the currently selected image.
         *
         * We want the new currentl selected image to be under the cursor in the
         * gallery, so we need to do some fanagling. The currently selected image
         * is set by its UUID, not its index in the image list.
         *
         * Get the currently selected image's index.
         */
        const imageToDeleteIndex = tempImages.findIndex(
          (image) => image.uuid === uuid
        );

        /**
         * New current image needs to be in the same spot, but because the gallery
         * is sorted in reverse order, the new current image's index will actuall be
         * one less than the deleted image's index.
         *
         * Clamp the new index to ensure it is valid..
         */
        const newCurrentImageIndex = clamp(
          imageToDeleteIndex,
          0,
          newImages.length - 1
        );

        state.currentImage = newImages.length
          ? newImages[newCurrentImageIndex]
          : undefined;

        state.currentImageUuid = newImages.length
          ? newImages[newCurrentImageIndex].uuid
          : '';
      }

      state.categories[category as GalleryCategory].images = newImages;
    },
    addImage: (
      state,
      action: PayloadAction<{
        image: InvokeAI.Image;
        category: GalleryCategory;
      }>
    ) => {
      const { image: newImage, category } = action.payload;
      const { uuid, url, mtime } = newImage;

      const tempCategory = state.categories[category as GalleryCategory];

      // Do not add duplicate images
      if (tempCategory.images.find((i) => i.url === url && i.mtime === mtime)) {
        return;
      }

      tempCategory.images.unshift(newImage);
      if (state.shouldAutoSwitchToNewImages) {
        state.currentImageUuid = uuid;
        state.currentImage = newImage;
        state.currentCategory = category;
      }
      state.intermediateImage = undefined;
      tempCategory.latest_mtime = mtime;
    },
    setIntermediateImage: (
      state,
      action: PayloadAction<
        InvokeAI.Image & { boundingBox?: IRect; generationMode?: InvokeTabName }
      >
    ) => {
      state.intermediateImage = action.payload;
    },
    clearIntermediateImage: (state) => {
      state.intermediateImage = undefined;
    },
    selectNextImage: (state) => {
      const { currentImage } = state;
      if (!currentImage) return;
      const tempImages =
        state.categories[currentImage.category as GalleryCategory].images;

      if (currentImage) {
        const currentImageIndex = tempImages.findIndex(
          (i) => i.uuid === currentImage.uuid
        );
        if (currentImageIndex < tempImages.length - 1) {
          const newCurrentImage = tempImages[currentImageIndex + 1];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }
      }
    },
    selectPrevImage: (state) => {
      const { currentImage } = state;
      if (!currentImage) return;
      const tempImages =
        state.categories[currentImage.category as GalleryCategory].images;

      if (currentImage) {
        const currentImageIndex = tempImages.findIndex(
          (i) => i.uuid === currentImage.uuid
        );
        if (currentImageIndex > 0) {
          const newCurrentImage = tempImages[currentImageIndex - 1];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }
      }
    },
    addGalleryImages: (state, action: PayloadAction<AddImagesPayload>) => {
      const { images, areMoreImagesAvailable, category } = action.payload;
      const tempImages = state.categories[category].images;

      // const prevImages = category === 'user' ? state.userImages : state.resultImages

      if (images.length > 0) {
        // Filter images that already exist in the gallery
        const newImages = images.filter(
          (newImage) =>
            !tempImages.find(
              (i) => i.url === newImage.url && i.mtime === newImage.mtime
            )
        );
        state.categories[category].images = tempImages
          .concat(newImages)
          .sort((a, b) => b.mtime - a.mtime);

        if (!state.currentImage) {
          const newCurrentImage = images[0];
          state.currentImage = newCurrentImage;
          state.currentImageUuid = newCurrentImage.uuid;
        }

        // keep track of the timestamps of latest and earliest images received
        state.categories[category].latest_mtime = images[0].mtime;
        state.categories[category].earliest_mtime =
          images[images.length - 1].mtime;
      }

      if (areMoreImagesAvailable !== undefined) {
        state.categories[category].areMoreImagesAvailable =
          areMoreImagesAvailable;
      }
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
    setCurrentCategory: (state, action: PayloadAction<GalleryCategory>) => {
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
  addImage,
  clearIntermediateImage,
  removeImage,
  setCurrentImage,
  addGalleryImages,
  setIntermediateImage,
  selectNextImage,
  selectPrevImage,
  setGalleryImageMinimumWidth,
  setGalleryImageObjectFit,
  setShouldAutoSwitchToNewImages,
  setCurrentCategory,
  setGalleryWidth,
  setShouldUseSingleGalleryColumn,
} = gallerySlice.actions;

export default gallerySlice.reducer;
