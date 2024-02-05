import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { uniqBy } from 'lodash-es';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import type { ImageDTO } from 'services/api/types';

import type { BoardId, GalleryState, GalleryView } from './types';
import { IMAGE_LIMIT, INITIAL_IMAGE_LIMIT } from './types';

export const initialGalleryState: GalleryState = {
  selection: [],
  shouldAutoSwitch: true,
  autoAssignBoardOnClick: true,
  autoAddBoardId: 'none',
  galleryImageMinimumWidth: 90,
  selectedBoardId: 'none',
  galleryView: 'images',
  boardSearchText: '',
  limit: INITIAL_IMAGE_LIMIT,
  offset: 0,
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
      state.limit = INITIAL_IMAGE_LIMIT;
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
      state.limit = INITIAL_IMAGE_LIMIT;
    },
    boardSearchTextChanged: (state, action: PayloadAction<string>) => {
      state.boardSearchText = action.payload;
    },
    moreImagesLoaded: (state) => {
      if (state.offset === 0 && state.limit === INITIAL_IMAGE_LIMIT) {
        state.offset = INITIAL_IMAGE_LIMIT;
        state.limit = IMAGE_LIMIT;
      } else {
        state.offset += IMAGE_LIMIT;
        state.limit += IMAGE_LIMIT;
      }
    },
  },
  extraReducers: (builder) => {
    builder.addMatcher(isAnyBoardDeleted, (state, action) => {
      const deletedBoardId = action.meta.arg.originalArgs;
      if (deletedBoardId === state.selectedBoardId) {
        state.selectedBoardId = 'none';
        state.galleryView = 'images';
      }
      if (deletedBoardId === state.autoAddBoardId) {
        state.autoAddBoardId = 'none';
      }
    });
    builder.addMatcher(boardsApi.endpoints.listAllBoards.matchFulfilled, (state, action) => {
      const boards = action.payload;
      if (!state.autoAddBoardId) {
        return;
      }

      if (!boards.map((b) => b.board_id).includes(state.autoAddBoardId)) {
        state.autoAddBoardId = 'none';
      }
    });
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
  moreImagesLoaded,
} = gallerySlice.actions;

const isAnyBoardDeleted = isAnyOf(
  imagesApi.endpoints.deleteBoard.matchFulfilled,
  imagesApi.endpoints.deleteBoardAndImages.matchFulfilled
);

export const selectGallerySlice = (state: RootState) => state.gallery;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateGalleryState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const galleryPersistConfig: PersistConfig<GalleryState> = {
  name: gallerySlice.name,
  initialState: initialGalleryState,
  migrate: migrateGalleryState,
  persistDenylist: ['selection', 'selectedBoardId', 'galleryView', 'offset', 'limit'],
};
