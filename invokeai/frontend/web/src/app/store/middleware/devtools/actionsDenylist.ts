/**
 * This is a list of actions that should be excluded in the Redux DevTools.
 */
export const actionsDenylist = [
  // very spammy canvas actions
  'canvas/setCursorPosition',
  'canvas/setStageCoordinates',
  'canvas/setStageScale',
  'canvas/setIsDrawing',
  'canvas/setBoundingBoxCoordinates',
  'canvas/setBoundingBoxDimensions',
  'canvas/setIsDrawing',
  'canvas/addPointToCurrentLine',
  // bazillions during generation
  'socket/socketGeneratorProgress',
  'socket/appSocketGeneratorProgress',
  // every time user presses shift
  // 'hotkeys/shiftKeyPressed',
  // this happens after every state change
  '@@REMEMBER_PERSISTED',
];
