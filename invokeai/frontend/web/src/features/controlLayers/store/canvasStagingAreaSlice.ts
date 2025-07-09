import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { PersistConfig, RootState } from 'app/store/store';
import { deepClone } from 'common/util/deepClone';
import { canvasReset } from 'features/controlLayers/store/actions';
import { queueApi } from 'services/api/endpoints/queue';

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
export const buildSelectSessionQueueItems = (sessionId: string) =>
  createSelector(
    [queueApi.endpoints.listAllQueueItems.select({ destination: sessionId }), selectDiscardedItems],
    ({ data }, discardedItems) => {
      if (!data) {
        return EMPTY_ARRAY;
      }
      return data.filter(
        ({ status, item_id }) => status !== 'canceled' && status !== 'failed' && !discardedItems.includes(item_id)
      );
    }
  );

export const selectIsStaging = (state: RootState) => {
  const sessionId = selectCanvasSessionId(state);
  const { data } = queueApi.endpoints.listAllQueueItems.select({ destination: sessionId })(state);
  if (!data) {
    return false;
  }
  const discardedItems = selectDiscardedItems(state);
  return data.some(
    ({ status, item_id }) => status !== 'canceled' && status !== 'failed' && !discardedItems.includes(item_id)
  );
};
const selectDiscardedItems = createSelector(
  selectCanvasSessionSlice,
  ({ canvasDiscardedQueueItems }) => canvasDiscardedQueueItems
);
