import { createSlice, isAnyOf,type PayloadAction } from '@reduxjs/toolkit';
import { migrateCanvasState } from 'app/store/migrations/canvasMigration';
import type { SliceConfig } from 'app/store/types';
import { canvasReset } from 'features/controlLayers/store/actions';
import { modelChanged } from 'features/controlLayers/store/paramsSlice';
import { calculateNewSize , getScaledBoundingBoxDimensions } from 'features/controlLayers/util/getScaledBoundingBoxDimensions';
import { isMainModelBase } from 'features/nodes/types/common';
import { API_BASE_MODELS } from 'features/parameters/types/constants';
import { getOptimalDimension } from 'features/parameters/util/optimalDimension';
import type { StateWithHistory } from 'redux-undo';
import { ActionCreators as UndoActionCreators } from 'redux-undo';
import { z } from 'zod';

import { instanceActions,undoableCanvasInstanceReducer } from './canvasInstanceSlice';
import type { CanvasState } from './types';
import { getInitialCanvasState } from './types';

interface CanvasesState {
  instances: Record<string, StateWithHistory<CanvasState>>;
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
      state.instances[canvasId] = undoableCanvasInstanceReducer(undefined, { type: '@@INIT' });
      
      // Set as active if it's the first instance
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
        state.activeInstanceId = remainingIds.length > 0 ? remainingIds[0]! : null;
      }
    },
    activeCanvasChanged: (state, action: PayloadAction<{ canvasId: string | null }>) => {
      state.activeInstanceId = action.payload.canvasId;
    },
    // Undo/Redo actions for active canvas
    canvasUndo: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId ?? state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.undo()
        );
      }
    },
    canvasRedo: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId ?? state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.redo()
        );
      }
    },
    canvasClearHistory: (state, action: PayloadAction<{ canvasId?: string }>) => {
      const canvasId = action.payload.canvasId ?? state.activeInstanceId;
      if (canvasId && state.instances[canvasId]) {
        state.instances[canvasId] = undoableCanvasInstanceReducer(
          state.instances[canvasId], 
          UndoActionCreators.clearHistory()
        );
      }
    },
  },
  extraReducers: (builder) => {
    // Forward all instanceActions to the correct canvas instance
    builder.addMatcher(
      isAnyOf(...Object.values(instanceActions)),
      (state, action) => {
        // Check if the action has a canvasId in the payload
        const actionWithCanvas = action as PayloadAction<{ canvasId?: string }>;
        const canvasId = actionWithCanvas.payload?.canvasId ?? state.activeInstanceId;
        
        if (canvasId && state.instances[canvasId]) {
          // Forward the action to the specific canvas instance (without the canvasId)
          const { canvasId: _, ...payloadWithoutCanvasId } = actionWithCanvas.payload || {};
          const forwardedAction = {
            ...action,
            payload: payloadWithoutCanvasId
          };
          
          state.instances[canvasId] = undoableCanvasInstanceReducer(
            state.instances[canvasId], 
            forwardedAction
          );
        }
      }
    );

    // Handle canvas reset for active canvas
    builder.addCase(canvasReset, (state) => {
      if (state.activeInstanceId && state.instances[state.activeInstanceId]) {
        const currentState = state.instances[state.activeInstanceId].present;
        const newState = getInitialCanvasState();
        
        // We need to retain the optimal dimension across resets, as it is changed only when the model changes.
        newState.bbox.modelBase = currentState.bbox.modelBase;
        const optimalDimension = getOptimalDimension(newState.bbox.modelBase);
        const rect = calculateNewSize(
          newState.bbox.aspectRatio.value,
          optimalDimension * optimalDimension,
          newState.bbox.modelBase
        );
        newState.bbox.rect.width = rect.width;
        newState.bbox.rect.height = rect.height;
        
        // Sync scaled size
        if (newState.bbox.scaleMethod === 'auto') {
          const { width, height } = newState.bbox.rect;
          newState.bbox.scaledSize = getScaledBoundingBoxDimensions({ width, height }, newState.bbox.modelBase);
        } else if (newState.bbox.scaleMethod === 'manual' && newState.bbox.aspectRatio.isLocked) {
          newState.bbox.scaledSize = calculateNewSize(
            newState.bbox.aspectRatio.value,
            newState.bbox.scaledSize.width * newState.bbox.scaledSize.height,
            newState.bbox.modelBase
          );
        }
        
        // Replace the current state with the reset state
        state.instances[state.activeInstanceId] = undoableCanvasInstanceReducer(undefined, { type: '@@INIT' });
        state.instances[state.activeInstanceId].present = newState;
      }
    });

    // Handle model changes for all canvas instances
    builder.addCase(modelChanged, (state, action) => {
      const { model } = action.payload;
      const base = model?.base;
      
      if (!isMainModelBase(base)) {
        return;
      }
      
      // Update all canvas instances when the model changes
      Object.keys(state.instances).forEach((canvasId) => {
        const canvasInstance = state.instances[canvasId];
        if (canvasInstance && canvasInstance.present.bbox.modelBase !== base) {
          const currentState = canvasInstance.present;
          const newState = { ...currentState };
          
          newState.bbox.modelBase = base;
          if (API_BASE_MODELS.includes(base)) {
            newState.bbox.aspectRatio.isLocked = true;
            newState.bbox.aspectRatio.value = 1;
            newState.bbox.aspectRatio.id = '1:1';
            newState.bbox.rect.width = 1024;
            newState.bbox.rect.height = 1024;
          }

          // Sync scaled size
          if (newState.bbox.scaleMethod === 'auto') {
            const { width, height } = newState.bbox.rect;
            newState.bbox.scaledSize = getScaledBoundingBoxDimensions({ width, height }, newState.bbox.modelBase);
          } else if (newState.bbox.scaleMethod === 'manual' && newState.bbox.aspectRatio.isLocked) {
            newState.bbox.scaledSize = calculateNewSize(
              newState.bbox.aspectRatio.value,
              newState.bbox.scaledSize.width * newState.bbox.scaledSize.height,
              newState.bbox.modelBase
            );
          }
          
          // Update the state
          state.instances[canvasId] = {
            ...canvasInstance,
            present: newState
          };
        }
      });
    });
  },
});

export const {
  canvasInstanceAdded,
  canvasInstanceRemoved,
  activeCanvasChanged,
  canvasUndo,
  canvasRedo,
  canvasClearHistory,
} = canvasesSlice.actions;

// Define schema for CanvasesState
const zCanvasesState = z.object({
  instances: z.record(z.string(), z.any()), // Undoable<CanvasState> is complex, using z.any()
  activeInstanceId: z.string().nullable(),
});

const getInitialCanvasesState = (): CanvasesState => initialCanvasesState;

export const canvasesSliceConfig: SliceConfig<typeof canvasesSlice> = {
  slice: canvasesSlice,
  schema: zCanvasesState,
  getInitialState: getInitialCanvasesState,
  persistConfig: {
    migrate: migrateCanvasState,
  },
};