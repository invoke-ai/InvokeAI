import type { PayloadAction } from '@reduxjs/toolkit';
import { createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';

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
