import { nanoid } from '@reduxjs/toolkit';
import type { StateWithHistory } from 'redux-undo';
import type { CanvasState } from 'features/controlLayers/store/types';
import { getInitialCanvasState } from 'features/controlLayers/store/types';

// Type alias for backward compatibility
type Undoable<T> = StateWithHistory<T>;

/**
 * Migration from single canvas (v1) to multi-canvas (v2) state structure.
 * 
 * This migration wraps the existing canvas state in the new instances structure:
 * - Creates a new canvases slice with instances dictionary
 * - Wraps the old canvas state in an Undoable structure
 * - Sets the first instance as active
 * 
 * Before: { canvas: { present: CanvasState, past: [], future: [] }, ... }
 * After: { canvases: { instances: { [id]: { present: CanvasState, past: [], future: [] } }, activeInstanceId: id }, ... }
 */
export const migrateCanvasV1ToV2 = (state: any) => {
  // Check if we have old canvas state but no canvases state
  if (state.canvas && !state.canvases) {
    const canvasId = nanoid();
    
    // The canvas state is already undoable, so we can use it directly
    const undoableState: Undoable<CanvasState> = state.canvas;
    
    // Remove the old canvas state and replace with new structure
    const { canvas: _oldCanvas, ...restState } = state;
    
    return {
      ...restState,
      canvases: {
        instances: {
          [canvasId]: undoableState
        },
        activeInstanceId: canvasId
      }
    };
  }
  
  return state;
};

/**
 * Migration helper to ensure backward compatibility.
 * This can be used in the persist config of the canvases slice.
 */
export const migrateCanvasState = (state: any, version?: number) => {
  // Apply v1 to v2 migration if needed
  let migratedState = migrateCanvasV1ToV2(state);
  
  // Future migrations can be added here
  // if (version < 3) {
  //   migratedState = migrateV2ToV3(migratedState);
  // }
  
  return migratedState;
};

/**
 * Helper to check if the state needs migration
 */
export const needsCanvasMigration = (state: any): boolean => {
  return Boolean(state.canvas && !state.canvases);
};

/**
 * Creates a default canvas instance for new installations
 */
export const createDefaultCanvasInstance = (): { instances: Record<string, Undoable<CanvasState>>, activeInstanceId: string } => {
  const canvasId = nanoid();
  return {
    instances: {
      [canvasId]: {
        past: [],
        present: getInitialCanvasState(), // This will need to be imported
        future: []
      }
    },
    activeInstanceId: canvasId
  };
};

// Note: getInitialCanvasState is now imported from types