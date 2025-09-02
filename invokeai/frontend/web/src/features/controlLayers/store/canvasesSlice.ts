import { createSlice, type PayloadAction, isAnyOf } from '@reduxjs/toolkit';
import { nanoid } from '@reduxjs/toolkit';
import type { SliceConfig } from 'app/store/types';
import { canvasReset } from 'features/controlLayers/store/actions';
import { undoableCanvasInstanceReducer, instanceActions } from './canvasInstanceSlice';
import type { Undoable } from 'redux-undo';
import { ActionCreators as UndoActionCreators } from 'redux-undo';
import type { CanvasState } from './types';
import { zCanvasState, getInitialCanvasState } from './types';
import { migrateCanvasState } from 'app/store/migrations/canvasMigration';

interface CanvasesState {
  instances: Record<string, Undoable<CanvasState>>;
  activeInstanceId: string | null;
}

const initialCanvasesState: CanvasesState = { 
  instances: {}, 
  activeInstanceId: null 
};

export const canvasesSlice = createSlice({
  name: 'canvases',
  initialState: initialCanvasesState,
  reducers: {
    canvasInstanceAdded: (state, action: PayloadAction<{ canvasId: string; name?: string }>) => {
      const { canvasId } = action.payload;
      // Initialize a new canvas instance with the undoable reducer
      state.instances[canvasId] = undoableCanvasInstanceReducer(undefined, { type: '@@INIT' });
      
      // If this is the first instance, make it active
      if (state.activeInstanceId === null) {
        state.activeInstanceId = canvasId;
      }
    },
    canvasInstanceRemoved: (state, action: PayloadAction<{ canvasId: string }>) => {
      const { canvasId } = action.payload;
      delete state.instances[canvasId];
      
      // If we removed the active instance, select another one
      if (state.activeInstanceId === canvasId) {
        const remainingIds = Object.keys(state.instances);
        state.activeInstanceId = remainingIds.length > 0 ? remainingIds[0] : null;
      }
    },
    activeCanvasChanged: (state, action: PayloadAction<{ canvasId: string | null }>) => {
      const { canvasId } = action.payload;
      if (canvasId && state.instances[canvasId]) {
        state.activeInstanceId = canvasId;
      } else {
        state.activeInstanceId = null;
      }
    },
    canvasInstanceRenamed: (state, action: PayloadAction<{ canvasId: string; name: string }>) => {
      const { canvasId, name } = action.payload;
      // Note: Canvas name could be stored in the instance metadata if needed
      // For now, this action exists for future UI implementation
    },
    // Undo/Redo actions for specific canvas instances
    canvasUndo: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId || state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.undo()
        );
      }
    },
    canvasRedo: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId || state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.redo()
        );
      }
    },
    canvasClearHistory: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId || state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.clearHistory()
        );
      }
    },
  },
  extraReducers: (builder) => {
    // Route all instance actions to the correct canvas instance
    builder.addMatcher(
      isAnyOf(...Object.values(instanceActions)),
      (state, action) => {
        // The action payload should contain canvasId for routing
        const canvasId = (action as PayloadAction<{ canvasId?: string }>).payload?.canvasId || state.activeInstanceId;
        if (canvasId && state.instances[canvasId]) {
          state.instances[canvasId] = undoableCanvasInstanceReducer(state.instances[canvasId], action);
        }
      }
    );

    // Handle canvas reset - reset the active canvas instance
    builder.addCase(canvasReset, (state) => {
      if (state.activeInstanceId && state.instances[state.activeInstanceId]) {
        // Reset the active canvas instance to initial state
        state.instances[state.activeInstanceId] = undoableCanvasInstanceReducer(undefined, { type: '@@INIT' });
      }
    });
  },
});

export const {
  canvasInstanceAdded,
  canvasInstanceRemoved,
  activeCanvasChanged,
  canvasInstanceRenamed,
  canvasUndo,
  canvasRedo,
  canvasClearHistory,
} = canvasesSlice.actions;

// Slice configuration for the store
export const canvasesSliceConfig: SliceConfig<typeof canvasesSlice> = {
  slice: canvasesSlice,
  getInitialState: () => {
    // For backward compatibility, create a default instance if none exist
    const defaultCanvasId = nanoid();
    const initialState: CanvasesState = {
      instances: {
        [defaultCanvasId]: undoableCanvasInstanceReducer(undefined, { type: '@@INIT' })
      },
      activeInstanceId: defaultCanvasId
    };
    return initialState;
  },
  persistConfig: {
    migrate: migrateCanvasState,
  },
};