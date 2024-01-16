/**
 * This is a list of actions that should be excluded in the Redux DevTools.
 */
export const actionsDenylist: string[] = [
  // very spammy canvas actions
  // 'canvas/setStageCoordinates',
  // 'canvas/setStageScale',
  // 'canvas/setBoundingBoxCoordinates',
  // 'canvas/setBoundingBoxDimensions',
  // 'canvas/addPointToCurrentLine',
  // bazillions during generation
  // 'socket/socketGeneratorProgress',
  // 'socket/appSocketGeneratorProgress',
  // this happens after every state change
  // '@@REMEMBER_PERSISTED',
];
