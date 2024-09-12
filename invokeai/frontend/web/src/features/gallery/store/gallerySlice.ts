import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { uniqBy } from 'lodash-es';
import type { ImageDTO } from 'services/api/types';

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
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageSelected: (state, action: PayloadAction<ImageDTO | null>) => {
      state.selection = action.payload ? [action.payload] : [];
    },
    selectionChanged: (state, action: PayloadAction<ImageDTO[]>) => {
      state.selection = uniqBy(action.payload, (i) => i.image_name);
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
