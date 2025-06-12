import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset } from 'features/controlLayers/store/actions';

type CanvasStagingAreaState = {
  generateSessionId: string | null;
  canvasSessionId: string | null;
};

const INITIAL_STATE: CanvasStagingAreaState = {
  generateSessionId: null,
  canvasSessionId: null,
};

const getInitialState = (): CanvasStagingAreaState => deepClone(INITIAL_STATE);

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
  reducers: {
    generateSessionIdCreated: {
      reducer: (state, action: PayloadAction<{ id: string }>) => {
        const { id } = action.payload;
        state.generateSessionId = id;
      },
      prepare: () => ({
        payload: { id: getPrefixedId('generate') },
      }),
    },
    generateSessionReset: (state) => {
      state.generateSessionId = null;
    },
    canvasSessionIdCreated: {
      reducer: (state, action: PayloadAction<{ id: string }>) => {
        const { id } = action.payload;
        state.canvasSessionId = id;
      },
      prepare: () => ({
        payload: { id: getPrefixedId('canvas') },
      }),
    },
    canvasSessionReset: (state) => {
      state.canvasSessionId = null;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, (state) => {
      state.canvasSessionId = null;
    });
  },
});

export const { generateSessionIdCreated, generateSessionReset, canvasSessionIdCreated, canvasSessionReset } =
  canvasSessionSlice.actions;

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
