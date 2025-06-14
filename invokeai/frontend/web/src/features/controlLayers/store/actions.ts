import { createAction } from '@reduxjs/toolkit';

// Needed to split this from canvasSlice.ts to avoid circular dependencies
export const canvasReset = createAction('canvas/canvasReset');
