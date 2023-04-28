import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import * as InvokeAI from 'app/invokeai';
import { invocationComplete } from 'services/events/actions';
import { InvokeTabName } from 'features/ui/store/tabMap';
import { IRect } from 'konva/lib/types';
import { clamp } from 'lodash';
import { isImageOutput } from 'services/types/guards';
import { deserializeImageResponse } from 'services/util/deserializeImageResponse';
import { imageUploaded } from 'services/thunks/image';

export type GalleryCategory = 'user' | 'result';

export type AddImagesPayload = {
  images: Array<InvokeAI._Image>;
  areMoreImagesAvailable: boolean;
  category: GalleryCategory;
};

type GalleryImageObjectFitType = 'contain' | 'cover';

export type Gallery = {
  images: InvokeAI._Image[];
  latest_mtime?: number;
  earliest_mtime?: number;
  areMoreImagesAvailable: boolean;
};

export interface GalleryState {
  /**
   * The selected image's unique name
   * Use `selectedImageSelector` to access the image
   */
  selectedImageName: string;
  /**
   * The currently selected image
   * @deprecated See `state.gallery.selectedImageName`
   */
  currentImage?: InvokeAI._Image;
  /**
   * The currently selected image's uuid.
   * @deprecated See `state.gallery.selectedImageName`, use `selectedImageSelector` to access the image
   */
  currentImageUuid: string;
  /**
   * The current progress image
   * @deprecated See `state.system.progressImage`
   */
  intermediateImage?: InvokeAI._Image & {
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
  selectedImageName: '',
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
    imageSelected: (state, action: PayloadAction<string>) => {
      state.selectedImageName = action.payload;
    },
    setCurrentImage: (state, action: PayloadAction<InvokeAI._Image>) => {
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
        image: InvokeAI._Image;
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
        InvokeAI._Image & {
          boundingBox?: IRect;
          generationMode?: InvokeTabName;
        }
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
  extraReducers(builder) {
    /**
     * Invocation Complete
     */
    builder.addCase(invocationComplete, (state, action) => {
      const { data } = action.payload;
      if (isImageOutput(data.result)) {
        state.selectedImageName = data.result.image.image_name;
        state.intermediateImage = undefined;
      }
    });

    /**
     * Upload Image - FULFILLED
     */
    builder.addCase(imageUploaded.fulfilled, (state, action) => {
      const { response } = action.payload;

      const uploadedImage = deserializeImageResponse(response);
      state.selectedImageName = uploadedImage.name;
    });
  },
});

export const {
  imageSelected,
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
