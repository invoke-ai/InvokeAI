import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { isEqual, uniqBy } from 'lodash-es';
import type { BoardRecordOrderBy, ImageDTO } from 'services/api/types';

import type { BoardId, ComparisonMode, GalleryState, GalleryView, OrderDir } from './types';

const initialGalleryState: GalleryState = {
  selection: [],
  shouldAutoSwitch: true,
  autoAssignBoardOnClick: true,
  autoAddBoardId: 'none',
  galleryImageMinimumWidth: 90,
  alwaysShowImageSizeBadge: false,
  selectedBoardId: 'none',
  galleryView: 'images',
  boardSearchText: '',
  limit: 20,
  offset: 0,
  starredFirst: true,
  orderDir: 'DESC',
  searchTerm: '',
  imageToCompare: null,
  comparisonMode: 'slider',
  comparisonFit: 'fill',
  shouldShowArchivedBoards: false,
  boardsListOrderBy: 'created_at',
  boardsListOrderDir: 'DESC',
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageSelected: (state, action: PayloadAction<ImageDTO | null>) => {
      // Let's be efficient here and not update the selection unless it has actually changed. This helps to prevent
      // unnecessary re-renders of the gallery.

      const selectedImage = action.payload;

      // If we got `null`, clear the selection
      if (!selectedImage) {
        // But only if we have images selected
        if (state.selection.length > 0) {
          state.selection = [];
        }
        return;
      }

      // If we have multiple images selected, clear the selection and select the new image
      if (state.selection.length !== 1) {
        state.selection = [selectedImage];
        return;
      }

      // If the selected image is different from the current selection, clear the selection and select the new image
      if (!isEqual(state.selection[0], selectedImage)) {
        state.selection = [selectedImage];
        return;
      }

      // Else we have the same image selected, do nothing
    },
    selectionChanged: (state, action: PayloadAction<ImageDTO[]>) => {
      // Let's be efficient here and not update the selection unless it has actually changed. This helps to prevent
      // unnecessary re-renders of the gallery.

      // Remove duplicates from the selection
      const newSelection = uniqBy(action.payload, (i) => i.image_name);

      // If the new selection has a different length, update the selection
      if (newSelection.length !== state.selection.length) {
        state.selection = newSelection;
        return;
      }

      // If the new selection is different, update the selection
      if (!isEqual(newSelection, state.selection)) {
        state.selection = newSelection;
        return;
      }

      // Else we have the same selection, do nothing
    },
    imageToCompareChanged: (state, action: PayloadAction<ImageDTO | null>) => {
      state.imageToCompare = action.payload;
    },
    comparisonModeChanged: (state, action: PayloadAction<ComparisonMode>) => {
      state.comparisonMode = action.payload;
    },
    comparisonModeCycled: (state) => {
      switch (state.comparisonMode) {
        case 'slider':
          state.comparisonMode = 'side-by-side';
          break;
        case 'side-by-side':
          state.comparisonMode = 'hover';
          break;
        case 'hover':
          state.comparisonMode = 'slider';
          break;
      }
    },
    shouldAutoSwitchChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldAutoSwitch = action.payload;
    },
    setGalleryImageMinimumWidth: (state, action: PayloadAction<number>) => {
      state.galleryImageMinimumWidth = action.payload;
    },
    autoAssignBoardOnClickChanged: (state, action: PayloadAction<boolean>) => {
      state.autoAssignBoardOnClick = action.payload;
    },
    boardIdSelected: (state, action: PayloadAction<{ boardId: BoardId; selectedImageName?: string }>) => {
      state.selectedBoardId = action.payload.boardId;
      state.galleryView = 'images';
      state.offset = 0;
    },
    autoAddBoardIdChanged: (state, action: PayloadAction<BoardId>) => {
      if (!action.payload) {
        state.autoAddBoardId = 'none';
        return;
      }
      state.autoAddBoardId = action.payload;
    },
    galleryViewChanged: (state, action: PayloadAction<GalleryView>) => {
      state.galleryView = action.payload;
      state.offset = 0;
    },
    boardSearchTextChanged: (state, action: PayloadAction<string>) => {
      state.boardSearchText = action.payload;
    },
    alwaysShowImageSizeBadgeChanged: (state, action: PayloadAction<boolean>) => {
      state.alwaysShowImageSizeBadge = action.payload;
    },
    comparedImagesSwapped: (state) => {
      if (state.imageToCompare) {
        const oldSelection = state.selection;
        state.selection = [state.imageToCompare];
        state.imageToCompare = oldSelection[0] ?? null;
      }
    },
    comparisonFitChanged: (state, action: PayloadAction<'contain' | 'fill'>) => {
      state.comparisonFit = action.payload;
    },
    offsetChanged: (state, action: PayloadAction<{ offset: number; withHotkey?: 'arrow' | 'alt+arrow' }>) => {
      const { offset } = action.payload;
      state.offset = offset;
    },
    limitChanged: (state, action: PayloadAction<number>) => {
      state.limit = action.payload;
    },
    shouldShowArchivedBoardsChanged: (state, action: PayloadAction<boolean>) => {
      state.shouldShowArchivedBoards = action.payload;
    },
    starredFirstChanged: (state, action: PayloadAction<boolean>) => {
      state.starredFirst = action.payload;
    },
    orderDirChanged: (state, action: PayloadAction<OrderDir>) => {
      state.orderDir = action.payload;
    },
    searchTermChanged: (state, action: PayloadAction<string>) => {
      state.searchTerm = action.payload;
      state.offset = 0;
    },
    boardsListOrderByChanged: (state, action: PayloadAction<BoardRecordOrderBy>) => {
      state.boardsListOrderBy = action.payload;
    },
    boardsListOrderDirChanged: (state, action: PayloadAction<OrderDir>) => {
      state.boardsListOrderDir = action.payload;
    },
  },
});

export const {
  imageSelected,
  shouldAutoSwitchChanged,
  autoAssignBoardOnClickChanged,
  setGalleryImageMinimumWidth,
  boardIdSelected,
  autoAddBoardIdChanged,
  galleryViewChanged,
  selectionChanged,
  boardSearchTextChanged,
  alwaysShowImageSizeBadgeChanged,
  imageToCompareChanged,
  comparisonModeChanged,
  comparedImagesSwapped,
  comparisonFitChanged,
  comparisonModeCycled,
  offsetChanged,
  limitChanged,
  orderDirChanged,
  starredFirstChanged,
  shouldShowArchivedBoardsChanged,
  searchTermChanged,
  boardsListOrderByChanged,
  boardsListOrderDirChanged,
} = gallerySlice.actions;

export const selectGallerySlice = (state: RootState) => state.gallery;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrateGalleryState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const galleryPersistConfig: PersistConfig<GalleryState> = {
  name: gallerySlice.name,
  initialState: initialGalleryState,
  migrate: migrateGalleryState,
  persistDenylist: ['selection', 'selectedBoardId', 'galleryView', 'offset', 'limit', 'imageToCompare'],
};
