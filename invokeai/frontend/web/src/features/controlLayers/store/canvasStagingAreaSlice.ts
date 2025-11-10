import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';

import { selectActiveCanvasId } from './selectors';
import type { CanvasStagingAreaState } from './types';

export const getInitialCanvasStagingAreaState = (): CanvasStagingAreaState => ({
  _version: 1,
  canvasSessionId: getPrefixedId('canvas'),
  canvasDiscardedQueueItems: [],
});

export const canvasStagingAreaState = createSlice({
  name: 'canvasSession',
  initialState: {} as CanvasStagingAreaState,
  reducers: {
    canvasQueueItemDiscarded: (state, action: PayloadAction<{ itemId: number }>) => {
      const { itemId } = action.payload;

      if (!state.canvasDiscardedQueueItems.includes(itemId)) {
        state.canvasDiscardedQueueItems.push(itemId);
      }
    },
    canvasSessionReset: {
      reducer: (state, action: PayloadAction<{ canvasSessionId: string }>) => {
        const { canvasSessionId } = action.payload;

        state.canvasSessionId = canvasSessionId;
        state.canvasDiscardedQueueItems = [];
      },
      prepare: () => {
        return {
          payload: {
            canvasSessionId: getPrefixedId('canvas'),
          },
        };
      },
    },
  },
});

export const isCanvasStagingAreaStateAction = isAnyOf(...Object.values(canvasStagingAreaState.actions));

export const { canvasSessionReset, canvasQueueItemDiscarded } = canvasStagingAreaState.actions;

export const selectCanvasStagingAreaByCanvasId = (state: RootState, canvasId: string) => {
  const instance = state.canvas.canvases[canvasId];
  assert(instance, 'Canvas does not exist');
  return instance.staging;
};
const selectCanvasStagingAreaBySessionId = (state: RootState, sessionId: string) => {
  const instance = Object.values(state.canvas.canvases).find((canvas) => canvas.staging.canvasSessionId === sessionId);
  assert(instance, 'Canvas does not exist');
  return instance.staging;
};
const selectActiveCanvasStagingArea = (state: RootState) => {
  const canvasId = selectActiveCanvasId(state);
  return selectCanvasStagingAreaByCanvasId(state, canvasId);
};
export const selectActiveCanvasStagingAreaSessionId = (state: RootState) => {
  const session = selectActiveCanvasStagingArea(state);
  return session.canvasSessionId;
};
const selectAllQueueItems = createSelector(
  (state: RootState) => state,
  (_state: RootState, sessionId: string) => sessionId,
  (state, sessionId) => queueApi.endpoints.listAllQueueItems.select({ destination: sessionId })(state)
);
const selectCanvasQueueItemsBySessionId = createSelector(
  selectAllQueueItems,
  (state: RootState, sessionId: string) => selectCanvasStagingAreaBySessionId(state, sessionId),
  ({ data }, session) => {
    if (!data) {
      return EMPTY_ARRAY;
    }
    return data.filter(
      ({ status, item_id }) =>
        status !== 'canceled' && status !== 'failed' && !session.canvasDiscardedQueueItems?.includes(item_id)
    );
  }
);
export const selectActiveCanvasQueueItems = (state: RootState) => {
  const session = selectActiveCanvasStagingArea(state);

  return selectCanvasQueueItemsBySessionId(state, session.canvasSessionId);
};
export const selectActiveCanvasIsStaging = (state: RootState) => {
  const queueItems = selectActiveCanvasQueueItems(state);

  return queueItems.length > 0;
};
