import { createAction, isAnyOf } from '@reduxjs/toolkit';

// Needed to split this from canvasSlice.ts to avoid circular dependencies
export const canvasReset = createAction('canvas/canvasReset');
export const newGallerySessionRequested = createAction('canvas/newGallerySessionRequested');
export const newCanvasSessionRequested = createAction('canvas/newCanvasSessionRequested');
export const newSessionRequested = isAnyOf(newGallerySessionRequested, newCanvasSessionRequested);
