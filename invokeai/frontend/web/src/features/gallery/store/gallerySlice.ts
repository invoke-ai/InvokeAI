import { objectEquals } from '@observ33r/object-equals';
import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { uniq } from 'lodash-es';
import type { BoardRecordOrderBy } from 'services/api/types';

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
    imageSelected: (state, action: PayloadAction<string | null>) => {
      // Let's be efficient here and not update the selection unless it has actually changed. This helps to prevent
      // unnecessary re-renders of the gallery.

      const selectedImageName = action.payload;

      // If we got `null`, clear the selection
      if (!selectedImageName) {
        // But only if we have images selected
        if (state.selection.length > 0) {
          state.selection = [];
        }
        return;
      }

      // If we have multiple images selected, clear the selection and select the new image
      if (state.selection.length !== 1) {
        state.selection = [selectedImageName];
        return;
      }

      // If the selected image is different from the current selection, clear the selection and select the new image
      if (state.selection[0] !== selectedImageName) {
        state.selection = [selectedImageName];
        return;
      }

      // Else we have the same image selected, do nothing
    },
    selectionChanged: (state, action: PayloadAction<string[]>) => {
      // Let's be efficient here and not update the selection unless it has actually changed. This helps to prevent
      // unnecessary re-renders of the gallery.

      // Remove duplicates from the selection
      const newSelection = uniq(action.payload);

      // If the new selection has a different length, update the selection
      if (newSelection.length !== state.selection.length) {
        state.selection = newSelection;
        return;
      }

      // If the new selection is different, update the selection
      if (!objectEquals(newSelection, state.selection)) {
        state.selection = newSelection;
        return;
      }

      // Else we have the same selection, do nothing
    },
    imageToCompareChanged: (state, action: PayloadAction<string | null>) => {
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
  persistDenylist: ['selection', 'selectedBoardId', 'galleryView', 'imageToCompare'],
};
