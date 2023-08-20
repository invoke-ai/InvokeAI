import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import { boardsApi } from 'services/api/endpoints/boards';
import { imagesApi } from 'services/api/endpoints/images';
import { ImageDTO } from 'services/api/types';
import { BoardId, GalleryState, GalleryView } from './types';

export const initialGalleryState: GalleryState = {
  selection: [],
  shouldAutoSwitch: true,
  autoAssignBoardOnClick: true,
  autoAddBoardId: 'none',
  galleryImageMinimumWidth: 96,
  selectedBoardId: 'none',
  galleryView: 'images',
  boardSearchText: '',
};

export const gallerySlice = createSlice({
  name: 'gallery',
  initialState: initialGalleryState,
  reducers: {
    imageSelected: (state, action: PayloadAction<ImageDTO | null>) => {
      state.selection = action.payload ? [action.payload] : [];
    },
    selectionChanged: (state, action: PayloadAction<ImageDTO[]>) => {
      state.selection = action.payload;
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
    boardIdSelected: (state, action: PayloadAction<BoardId>) => {
      state.selectedBoardId = action.payload;
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
    builder.addMatcher(
      boardsApi.endpoints.listAllBoards.matchFulfilled,
      (state, action) => {
        const boards = action.payload;
        if (!state.autoAddBoardId) {
          return;
        }

        if (!boards.map((b) => b.board_id).includes(state.autoAddBoardId)) {
          state.autoAddBoardId = 'none';
        }
      }
    );
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
} = gallerySlice.actions;

export default gallerySlice.reducer;

const isAnyBoardDeleted = isAnyOf(
  imagesApi.endpoints.deleteBoard.matchFulfilled,
  imagesApi.endpoints.deleteBoardAndImages.matchFulfilled
);
