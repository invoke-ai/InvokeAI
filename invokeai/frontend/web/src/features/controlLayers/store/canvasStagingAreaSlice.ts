import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { useMemo } from 'react';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';
import z from 'zod';

const zCanvasStagingAreaState = z.object({
  _version: z.literal(1),
  canvasSessionId: z.string(),
  canvasDiscardedQueueItems: z.array(z.number().int()),
});
type CanvasStagingAreaState = z.infer<typeof zCanvasStagingAreaState>;

const getInitialState = (): CanvasStagingAreaState => ({
  _version: 1,
  canvasSessionId: getPrefixedId('canvas'),
  canvasDiscardedQueueItems: [],
});

const slice = createSlice({
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

export const { canvasSessionReset, canvasQueueItemDiscarded } = slice.actions;

export const canvasSessionSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasStagingAreaState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        state._version = 1;
        state.canvasSessionId = state.canvasSessionId ?? getPrefixedId('canvas');
      }

      return zCanvasStagingAreaState.parse(state);
    },
  },
};

export const selectCanvasSessionSlice = (s: RootState) => s[slice.name];
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
