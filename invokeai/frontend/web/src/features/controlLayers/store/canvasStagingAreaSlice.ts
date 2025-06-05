import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { canvasReset } from 'features/controlLayers/store/actions';

type CanvasStagingAreaState = {
  type: 'simple' | 'advanced';
  id: string | null;
};

const INITIAL_STATE: CanvasStagingAreaState = {
  type: 'simple',
  id: null,
};

const getInitialState = (): CanvasStagingAreaState => deepClone(INITIAL_STATE);

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
  reducers: {
    canvasSessionTypeChanged: (state, action: PayloadAction<{ type: CanvasStagingAreaState['type'] }>) => {
      const { type } = action.payload;
      state.type = type;
      state.id = null;
    },
    canvasSessionGenerationStarted: {
      reducer: (state, action: PayloadAction<{ id: string }>) => {
        const { id } = action.payload;
        state.id = id;
      },
      prepare: () => ({
        payload: { id: getPrefixedId('canvas') },
      }),
    },
    canvasSessionGenerationFinished: (state) => {
      state.id = null;
    },
    canvasSessionReset: () => getInitialState(),
  },
  extraReducers(builder) {
    builder.addCase(canvasReset, () => getInitialState());
  },
});

export const {
  canvasSessionTypeChanged,
  canvasSessionGenerationStarted,
  canvasSessionReset,
  canvasSessionGenerationFinished,
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

export const selectIsStaging = createSelector(selectCanvasSessionSlice, ({ id }) => id !== null);
export const selectCanvasSessionType = createSelector(selectCanvasSessionSlice, ({ type }) => type);
export const selectCanvasSessionId = createSelector(selectCanvasSessionSlice, ({ id }) => id);
