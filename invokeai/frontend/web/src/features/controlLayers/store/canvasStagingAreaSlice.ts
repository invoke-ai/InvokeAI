import type { PayloadAction } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
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

export const { canvasSessionReset, canvasQueueItemDiscarded } = canvasStagingAreaState.actions;

const findCanvasStagingAreaByCanvasId = (state: RootState, canvasId: string) => {
  const instance = state.canvas.canvases[canvasId];
  assert(instance, 'Canvas does not exist');
  return instance.staging;
};
export const selectCanvasStagingAreaByCanvasId = (state: RootState, canvasId: string) =>
  findCanvasStagingAreaByCanvasId(state, canvasId);
const selectActiveCanvasStagingArea = (state: RootState) => {
  const canvasId = selectActiveCanvasId(state);
  return findCanvasStagingAreaByCanvasId(state, canvasId);
};
const selectCanvasStagingAreaBySessionId = (state: RootState, sessionId: string) => {
  const instance = Object.values(state.canvas.canvases).find((canvas) => canvas.staging.canvasSessionId === sessionId);
  assert(instance, 'Canvas does not exist');
  return instance.staging;
};
export const selectCanvasStagingAreaSessionId = (state: RootState, canvasId: string) => {
  const session = selectCanvasStagingAreaByCanvasId(state, canvasId);
  return session.canvasSessionId;
};
export const selectActiveCanvasStagingAreaSessionId = (state: RootState) => {
  const session = selectActiveCanvasStagingArea(state);
  return session.canvasSessionId;
};
const selectCanvasStagingAreaDiscardedItemsBySessionId = (state: RootState, sessionId: string) => {
  const session = selectCanvasStagingAreaBySessionId(state, sessionId);
  return session.canvasDiscardedQueueItems;
};
export const buildSelectCanvasQueueItemsBySessionId = (sessionId: string) =>
  createSelector(
    queueApi.endpoints.listAllQueueItems.select({ destination: sessionId }),
    (state: RootState) => selectCanvasStagingAreaDiscardedItemsBySessionId(state, sessionId),
    ({ data }, discardedItems) => {
      if (!data) {
        return EMPTY_ARRAY;
      }
      return data.filter(
        ({ status, item_id }) => status !== 'canceled' && status !== 'failed' && !discardedItems?.includes(item_id)
      );
    }
  );
export const buildSelectIsStagingBySessionId = (sessionId: string) =>
  createSelector(buildSelectCanvasQueueItemsBySessionId(sessionId), (queueItems) => {
    return queueItems.length > 0;
  });
