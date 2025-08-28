import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { uniq } from 'es-toolkit/compat';
import type { BoardRecordOrderBy } from 'services/api/types';
import { assert } from 'tsafe';

import {
  type BoardId,
  type ComparisonMode,
  type GalleryState,
  type GalleryView,
  type OrderDir,
  zGalleryState,
} from './types';

const getInitialState = (): GalleryState => ({
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
});

const slice = createSlice({
  name: 'gallery',
  initialState: getInitialState(),
  reducers: {
    imageSelected: (state, action: PayloadAction<string | null>) => {
      const selectedImageName = action.payload;

      if (!selectedImageName) {
        state.selection = [];
      } else {
        state.selection = [selectedImageName];
      }
    },
    selectionChanged: (state, action: PayloadAction<string[]>) => {
      state.selection = uniq(action.payload);
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
      const { boardId, selectedImageName } = action.payload;
      state.selectedBoardId = boardId;
      state.galleryView = 'images';
      if (selectedImageName) {
        state.selection = [selectedImageName];
      }
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
} = slice.actions;

export const selectGallerySlice = (state: RootState) => state.gallery;

export const gallerySliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zGalleryState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
      }
      return zGalleryState.parse(state);
    },
    persistDenylist: ['selection', 'selectedBoardId', 'galleryView', 'imageToCompare'],
  },
};
