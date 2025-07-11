import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { PersistConfig, RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { deepClone } from 'common/util/deepClone';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useMemo } from 'react';
import { queueApi } from 'services/api/endpoints/queue';

type CanvasStagingAreaState = {
  _version: 1;
  canvasSessionId: string;
  canvasDiscardedQueueItems: number[];
};

const INITIAL_STATE: CanvasStagingAreaState = {
  _version: 1,
  canvasSessionId: getPrefixedId('canvas'),
  canvasDiscardedQueueItems: [],
};

const getInitialState = (): CanvasStagingAreaState => deepClone(INITIAL_STATE);

export const canvasSessionSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
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

export const { canvasSessionReset, canvasQueueItemDiscarded } = canvasSessionSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  if (!('_version' in state)) {
    state._version = 1;
    state.canvasSessionId = state.canvasSessionId ?? getPrefixedId('canvas');
  }

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

const selectDiscardedItems = createSelector(
  selectCanvasSessionSlice,
  ({ canvasDiscardedQueueItems }) => canvasDiscardedQueueItems
);

export const buildSelectCanvasQueueItems = (sessionId: string) =>
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

export const buildSelectIsStaging = (sessionId: string) =>
  createSelector([buildSelectCanvasQueueItems(sessionId)], (queueItems) => {
    return queueItems.length > 0;
  });
export const useCanvasIsStaging = () => {
  const sessionId = useAppSelector(selectCanvasSessionId);
  const selector = useMemo(() => buildSelectIsStaging(sessionId), [sessionId]);
  return useAppSelector(selector);
};
