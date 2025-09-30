import type { PayloadAction, UnknownAction } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import { EMPTY_ARRAY } from 'app/store/constants';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { queueApi } from 'services/api/endpoints/queue';
import { assert } from 'tsafe';
import z from 'zod';

import {
  canvasAdding,
  canvasDeleted,
  canvasInitialized,
  canvasMultiCanvasMigrated,
  MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
} from './canvasSlice';
import { selectActiveCanvasId } from './selectors';

const zCanvasSession = z.object({
  canvasId: z.string().min(1),
  canvasSessionId: z.string(),
  canvasDiscardedQueueItems: z.array(z.number().int()),
});
type CanvasSession = z.infer<typeof zCanvasSession>;
const zCanvasStagingAreaState = z.object({
  _version: z.literal(2),
  sessions: z.record(z.string(), zCanvasSession),
});
type CanvasStagingAreaState = z.infer<typeof zCanvasStagingAreaState>;

type CanvasPayload<T> = { canvasId: string } & T;
type CanvasPayloadAction<T> = PayloadAction<CanvasPayload<T>>;

const getInitialCanvasSessionState = (canvasId: string): CanvasSession => ({
  canvasId,
  canvasSessionId: getPrefixedId('canvas'),
  canvasDiscardedQueueItems: [],
});

const getInitialCanvasStagingAreaState = (): CanvasStagingAreaState => ({
  _version: 2,
  sessions: {},
});

const canvasStagingAreaSlice = createSlice({
  name: 'canvasSession',
  initialState: getInitialCanvasStagingAreaState,
  reducers: {},
  extraReducers(builder) {
    builder.addCase(canvasAdding, (state, action) => {
      const session = getInitialCanvasSessionState(action.payload.canvasId);
      state.sessions[session.canvasId] = session;
    });
    builder.addCase(canvasDeleted, (state, action) => {
      delete state.sessions[action.payload.canvasId];
    });
    builder.addCase(canvasMultiCanvasMigrated, (state, action) => {
      const session = state.sessions[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
      if (!session) {
        return;
      }
      session.canvasId = action.payload.canvasId;
      state.sessions[session.canvasId] = session;
      delete state.sessions[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
    });
    builder.addCase(canvasInitialized, (state, action) => {
      const canvasId = action.payload.canvasId;
      if (!state.sessions[canvasId]) {
        state.sessions[canvasId] = getInitialCanvasSessionState(canvasId);
      }
    });
  },
});

const canvasSessionFragment = createSlice({
  name: 'canvasSession',
  initialState: {} as CanvasSession,
  reducers: {
    canvasQueueItemDiscarded: (state, action: CanvasPayloadAction<{ itemId: number }>) => {
      const { itemId } = action.payload;

      if (!state.canvasDiscardedQueueItems.includes(itemId)) {
        state.canvasDiscardedQueueItems.push(itemId);
      }
    },
    canvasSessionReset: {
      reducer: (state, action: CanvasPayloadAction<{ canvasSessionId: string }>) => {
        const { canvasSessionId } = action.payload;

        state.canvasSessionId = canvasSessionId;
        state.canvasDiscardedQueueItems = [];
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
});

export const { canvasSessionReset, canvasQueueItemDiscarded } = canvasSessionFragment.actions;

const isCanvasSessionAction = isAnyOf(...Object.values(canvasSessionFragment.actions));

export const canvasSessionReducer = (
  state: CanvasStagingAreaState | undefined,
  action: UnknownAction
): CanvasStagingAreaState => {
  state = canvasStagingAreaSlice.reducer(state, action);

  if (!isCanvasSessionAction(action)) {
    return state;
  }

  const canvasId = action.payload.canvasId;

  return {
    ...state,
    sessions: {
      ...state.sessions,
      [canvasId]: canvasSessionFragment.reducer(state.sessions[canvasId], action),
    },
  };
};

export const canvasSessionSliceConfig: SliceConfig<typeof canvasStagingAreaSlice> = {
  slice: canvasStagingAreaSlice,
  schema: zCanvasStagingAreaState,
  getInitialState: getInitialCanvasStagingAreaState,
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
        } as CanvasSession;

        state = {
          _version: 2,
          sessions: { [session.canvasId]: session },
        };
      }

      return zCanvasStagingAreaState.parse(state);
    },
  },
};

const findSessionByCanvasId = (sessions: Record<string, CanvasSession>, canvasId: string) => {
  const session = sessions[canvasId];
  assert(session, 'Session must exist for a canvas once the canvas has been created');
  return session;
};
export const selectCanvasSessionByCanvasId = (state: RootState, canvasId: string) =>
  findSessionByCanvasId(state.canvasSession.sessions, canvasId);
const selectActiveCanvasSession = (state: RootState) => {
  const canvasId = selectActiveCanvasId(state);
  return findSessionByCanvasId(state.canvasSession.sessions, canvasId);
};
const selectCanvasSessionBySessionId = (state: RootState, sessionId: string) => {
  const session = Object.values(state.canvasSession.sessions).find((s) => s.canvasSessionId === sessionId);
  assert(session, 'Session does not exist');
  return session;
};
export const selectCanvasSessionId = (state: RootState, canvasId: string) => {
  const session = selectCanvasSessionByCanvasId(state, canvasId);
  return session.canvasSessionId;
};
export const selectActiveCanvasSessionId = (state: RootState) => {
  const session = selectActiveCanvasSession(state);
  return session.canvasSessionId;
};
const selectCanvasSessionDiscardedItemsBySessionId = (state: RootState, sessionId: string) => {
  const session = selectCanvasSessionBySessionId(state, sessionId);
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
