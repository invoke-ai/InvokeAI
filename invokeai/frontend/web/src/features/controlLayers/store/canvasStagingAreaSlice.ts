import { createSelector, createSlice, type PayloadAction } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';
import z from 'zod';

import {
  canvasCreated,
  canvasMultiCanvasMigrated,
  canvasRemoved,
  MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
} from './canvasSlice';
import { selectSelectedCanvasId } from './selectors';

const zCanvasSessionState = z.object({
  canvasId: z.string(),
  canvasSessionId: z.string(),
  canvasDiscardedQueueItems: z.array(z.number().int()),
});
type CanvasSessionState = z.infer<typeof zCanvasSessionState>;
const zCanvasStagingAreaState = z.object({
  _version: z.literal(2),
  sessions: z.array(zCanvasSessionState),
});
type CanvasStagingAreaState = z.infer<typeof zCanvasStagingAreaState>;

type CanvasPayload<T> = { canvasId: string } & T;
type CanvasPayloadAction<T> = PayloadAction<CanvasPayload<T>>;

const getInitialCanvasSessionState = (canvasId: string): CanvasSessionState => ({
  canvasId,
  canvasSessionId: getPrefixedId('canvas'),
  canvasDiscardedQueueItems: [],
});

const getInitialState = (): CanvasStagingAreaState => ({
  _version: 2,
  sessions: [],
});

const slice = createSlice({
  name: 'canvasSession',
  initialState: getInitialState(),
  reducers: {
    canvasQueueItemDiscarded: (state, action: CanvasPayloadAction<{ itemId: number }>) => {
      const { canvasId, itemId } = action.payload;

      const session = state.sessions.find((session) => session.canvasId === canvasId);
      if (!session) {
        return;
      }

      if (!session.canvasDiscardedQueueItems.includes(itemId)) {
        session.canvasDiscardedQueueItems.push(itemId);
      }
    },
    canvasSessionReset: {
      reducer: (state, action: CanvasPayloadAction<{ canvasSessionId: string }>) => {
        const { canvasId, canvasSessionId } = action.payload;

        const session = state.sessions.find((session) => session.canvasId === canvasId);
        if (!session) {
          return;
        }

        session.canvasSessionId = canvasSessionId;
        session.canvasDiscardedQueueItems = [];
      },
      prepare: (payload: CanvasPayload<object>) => {
        return {
          payload: {
            ...payload,
            canvasSessionId: getPrefixedId('canvas'),
          },
        };
      },
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasCreated, (state, action) => {
      const session = getInitialCanvasSessionState(action.payload.id);
      state.sessions.push(session);
    });
    builder.addCase(canvasRemoved, (state, action) => {
      state.sessions = state.sessions.filter((session) => session.canvasId !== action.payload.id);
    });
    builder.addCase(canvasMultiCanvasMigrated, (state, action) => {
      const session = state.sessions.find((session) => session.canvasId === MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER);
      if (!session) {
        return;
      }
      session.canvasId = action.payload.id;
    });
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
      } else if (state._version === 1) {
        // Migrate from v1 to v2: slice represented a canvas session instance -> slice represents multiple canvas session instances
        const session = {
          canvasId: MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
          ...state,
        } as CanvasSessionState;

        state = {
          _version: 2,
          sessions: [session],
        };
      }

      return zCanvasStagingAreaState.parse(state);
    },
  },
};

const findSessionByCanvasId = (sessions: CanvasSessionState[], canvasId: string) => {
  const session = sessions.find((s) => s.canvasId === canvasId);
  assert(session, 'Session must exist for a canvas once the canvas has been created');
  return session;
};
export const selectCanvasSessionByCanvasId = (state: RootState, canvasId: string) =>
  findSessionByCanvasId(state.canvasSession.sessions, canvasId);
const selectSelectedCanvasSession = (state: RootState) => {
  const canvasId = selectSelectedCanvasId(state);
  return findSessionByCanvasId(state.canvasSession.sessions, canvasId);
};
export const selectCanvasSessionId = (state: RootState, canvasId: string) => {
  const session = selectCanvasSessionByCanvasId(state, canvasId);
  return session.canvasSessionId;
};
export const selectSelectedCanvasSessionId = (state: RootState) => {
  const session = selectSelectedCanvasSession(state);
  return session.canvasSessionId;
};
const selectCanvasSessionDiscardedItemsBySessionId = (state: RootState, sessionId: string) => {
  const session = state.canvasSession.sessions.find((s) => s.canvasSessionId === sessionId);
  assert(session, 'Session does not exist');
  return session.canvasDiscardedQueueItems;
};
export const buildSelectCanvasQueueItemsBySessionId = (sessionId: string) =>
  createSelector(
    queueApi.endpoints.listAllQueueItems.select({ destination: sessionId }),
    (state: RootState) => selectCanvasSessionDiscardedItemsBySessionId(state, sessionId),
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
