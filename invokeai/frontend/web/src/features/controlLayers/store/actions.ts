import { createAction, isAnyOf } from '@reduxjs/toolkit';

// Needed to split this from canvasSlice.ts to avoid circular dependencies
export const canvasReset = createAction('canvas/canvasReset');
export const newSimpleCanvasSessionRequested = createAction('canvas/newSimpleCanvasSessionRequested');
export const newAdvancedCanvasSessionRequested = createAction('canvas/newAdvancedCanvasSessionRequested');
export const newSessionRequested = isAnyOf(newSimpleCanvasSessionRequested, newAdvancedCanvasSessionRequested);
