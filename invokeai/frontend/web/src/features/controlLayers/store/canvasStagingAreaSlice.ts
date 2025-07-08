import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset } from 'features/controlLayers/store/actions';

type CanvasStagingAreaState = {
  generateSessionId: string | null;
  canvasSessionId: string | null;
  canvasDiscardedQueueItems: number[];
};

const INITIAL_STATE: CanvasStagingAreaState = {
  generateSessionId: null,
  canvasSessionId: null,
  canvasDiscardedQueueItems: [],
};

const getInitialState = (): CanvasStagingAreaState => deepClone(INITIAL_STATE);

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
  reducers: {
    generateSessionIdChanged: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.generateSessionId = id;
    },
    generateSessionReset: (state) => {
      state.generateSessionId = null;
    },
    canvasQueueItemDiscarded: (state, action: PayloadAction<{ itemId: number }>) => {
      const { itemId } = action.payload;
      if (!state.canvasDiscardedQueueItems.includes(itemId)) {
        state.canvasDiscardedQueueItems.push(itemId);
      }
    },
    canvasSessionIdChanged: (state, action: PayloadAction<{ id: string }>) => {
      const { id } = action.payload;
      state.canvasSessionId = id;
      state.canvasDiscardedQueueItems = [];
    },
    canvasSessionReset: (state) => {
      state.canvasSessionId = null;
      state.canvasDiscardedQueueItems = [];
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, (state) => {
      state.canvasSessionId = null;
    });
  },
});

export const {
  generateSessionIdChanged,
  generateSessionReset,
  canvasSessionIdChanged,
  canvasSessionReset,
  canvasQueueItemDiscarded,
} = canvasSessionSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasStagingAreaPersistConfig: PersistConfig<CanvasStagingAreaState> = {
  name: canvasSessionSlice.name,
  initialState: getInitialState(),
  migrate,
  persistDenylist: [],
};

export const selectCanvasSessionSlice = (s: RootState) => s[canvasSessionSlice.name];

export const selectCanvasSessionId = createSelector(selectCanvasSessionSlice, ({ canvasSessionId }) => canvasSessionId);
export const selectGenerateSessionId = createSelector(
  selectCanvasSessionSlice,
  ({ generateSessionId }) => generateSessionId
);
export const selectIsStaging = createSelector(selectCanvasSessionId, (canvasSessionId) => canvasSessionId !== null);
export const selectDiscardedItems = createSelector(
  selectCanvasSessionSlice,
  ({ canvasDiscardedQueueItems }) => canvasDiscardedQueueItems
);
