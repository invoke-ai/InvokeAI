import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { galleryImageClicked } from 'features/gallery/store/gallerySlice';

export type ViewerMode = 'image' | 'info' | 'progress';

export type ViewerState = {
  _version: 1;
  /**
   * The currently-selected viewer mode.
   */
  viewerMode: ViewerMode;
};

export const initialViewerState: ViewerState = {
  _version: 1,
  viewerMode: 'image',
};

export const viewerSlice = createSlice({
  name: 'viewer',
  initialState: initialViewerState,
  reducers: {
    viewerModeChanged: (state, action: PayloadAction<ViewerMode>) => {
      state.viewerMode = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(galleryImageClicked, (state) => {
      // When a gallery image is clicked and we are in progress mode, switch to image mode.
      // This is not the same as the gallery _selection_ being changed.
      if (state.viewerMode === 'progress') {
        state.viewerMode = 'image';
      }
    });
  },
});

export const { viewerModeChanged } = viewerSlice.actions;

export const selectViewerSlice = (state: RootState) => state.viewer;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
export const migrateViewerState = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
  }
  return state;
};

export const viewerPersistConfig: PersistConfig<ViewerState> = {
  name: viewerSlice.name,
  initialState: initialViewerState,
  migrate: migrateViewerState,
  persistDenylist: [],
};
