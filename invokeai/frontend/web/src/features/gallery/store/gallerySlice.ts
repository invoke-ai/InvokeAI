import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
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
    itemSelected: (state, action: PayloadAction<{ type: 'image' | 'video'; id: string } | null>) => {
      const selectedItem = action.payload;

      if (!selectedItem) {
        state.selection = [];
      } else {
        state.selection = [selectedItem];
      }
    },
    selectionChanged: (state, action: PayloadAction<{ type: 'image' | 'video'; id: string }[]>) => {
      const uniqueById = new Map<string, { type: 'image' | 'video'; id: string }>();
      for (const item of action.payload) {
        if (!uniqueById.has(item.id)) {
          uniqueById.set(item.id, item);
        }
      }
      state.selection = Array.from(uniqueById.values());
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
    boardIdSelected: (
      state,
      action: PayloadAction<{
        boardId: BoardId;
        select?: {
          selection: GalleryState['selection'];
          galleryView: GalleryState['galleryView'];
        };
      }>
    ) => {
      const { boardId, select } = action.payload;
      state.selectedBoardId = boardId;
      if (select) {
        state.selection = select.selection;
        state.galleryView = select.galleryView;
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
        state.selection = [{ type: 'image', id: state.imageToCompare }];
        state.imageToCompare = oldSelection[0]?.id ?? null;
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
  itemSelected,
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
