import { createSelector } from '@reduxjs/toolkit';
import { RootState } from 'app/store/store';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import { clamp, keyBy } from 'lodash-es';
import { ImageDTO } from 'services/api/types';
import {
  ASSETS_CATEGORIES,
  BoardId,
  IMAGE_CATEGORIES,
  INITIAL_IMAGE_LIMIT,
  imagesAdapter,
  initialGalleryState,
} from './gallerySlice';
import { ListImagesArgs } from 'services/api/endpoints/images';

export const gallerySelector = (state: RootState) => state.gallery;

const isInSelectedBoard = (
  selectedBoardId: BoardId,
  imageDTO: ImageDTO,
  batchImageNames: string[]
) => {
  if (selectedBoardId === 'all') {
    // all images are in the "All Images" board
    return true;
  }

  if (selectedBoardId === 'none' && !imageDTO.board_id) {
    // Only images without a board are in the "No Board" board
    return true;
  }

  if (
    selectedBoardId === 'batch' &&
    batchImageNames.includes(imageDTO.image_name)
  ) {
    // Only images with is_batch are in the "Batch" board
    return true;
  }

  return selectedBoardId === imageDTO.board_id;
};

export const selectFilteredImagesLocal = createSelector(
  [(state: typeof initialGalleryState) => state],
  (galleryState) => {
    const allImages = imagesAdapter.getSelectors().selectAll(galleryState);
    const { galleryView, selectedBoardId } = galleryState;

    const categories =
      galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES;

    const filteredImages = allImages.filter((i) => {
      const isInCategory = categories.includes(i.image_category);

      const isInBoard = isInSelectedBoard(
        selectedBoardId,
        i,
        galleryState.batchImageNames
      );
      return isInCategory && isInBoard;
    });

    return filteredImages;
  }
);

export const selectFilteredImages = createSelector(
  (state: RootState) => state,
  (state) => {
    return selectFilteredImagesLocal(state.gallery);
  },
  defaultSelectorOptions
);

export const selectFilteredImagesAsObject = createSelector(
  selectFilteredImages,
  (filteredImages) => keyBy(filteredImages, 'image_name')
);

export const selectFilteredImagesIds = createSelector(
  selectFilteredImages,
  (filteredImages) => filteredImages.map((i) => i.image_name)
);

export const selectLastSelectedImage = createSelector(
  (state: RootState) => state,
  (state) => state.gallery.selection[state.gallery.selection.length - 1],
  defaultSelectorOptions
);

export const selectSelectedImages = createSelector(
  (state: RootState) => state,
  (state) =>
    imagesAdapter
      .getSelectors()
      .selectAll(state.gallery)
      .filter((i) => state.gallery.selection.includes(i.image_name)),
  defaultSelectorOptions
);

export const selectNextImageToSelectLocal = createSelector(
  [
    (state: typeof initialGalleryState) => state,
    (state: typeof initialGalleryState, image_name: string) => image_name,
  ],
  (state, image_name) => {
    const filteredImages = selectFilteredImagesLocal(state);
    const ids = filteredImages.map((i) => i.image_name);

    const deletedImageIndex = ids.findIndex(
      (result) => result.toString() === image_name
    );

    const filteredIds = ids.filter((id) => id.toString() !== image_name);

    const newSelectedImageIndex = clamp(
      deletedImageIndex,
      0,
      filteredIds.length - 1
    );

    const newSelectedImageId = filteredIds[newSelectedImageIndex];

    return newSelectedImageId;
  }
);

export const selectNextImageToSelect = createSelector(
  [
    (state: RootState) => state,
    (state: RootState, image_name: string) => image_name,
  ],
  (state, image_name) => {
    return selectNextImageToSelectLocal(state.gallery, image_name);
  },
  defaultSelectorOptions
);

export const selectListImagesBaseQueryArgs = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { selectedBoardId, galleryView } = state.gallery;

    const listImagesBaseQueryArgs: ListImagesArgs = {
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      board_id: selectedBoardId === 'all' ? undefined : selectedBoardId,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  },
  defaultSelectorOptions
);

export const selectListAllImagesBaseQueryArgs = createSelector(
  [(state: RootState) => state],
  (state) => {
    const { galleryView } = state.gallery;

    const listImagesBaseQueryArgs = {
      categories:
        galleryView === 'images' ? IMAGE_CATEGORIES : ASSETS_CATEGORIES,
      offset: 0,
      limit: INITIAL_IMAGE_LIMIT,
      is_intermediate: false,
    };

    return listImagesBaseQueryArgs;
  },
  defaultSelectorOptions
);
