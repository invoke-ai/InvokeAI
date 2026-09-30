import type { Middleware } from '@reduxjs/toolkit';
import { entitySelected } from 'features/controlLayers/store/canvasSlice';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';

// Defer explicit layer selection before it reaches the reducer, preserving the edit session on Cancel.
export const vectorEditSelectionMiddleware: Middleware = () => (next) => (action) => {
  if (entitySelected.match(action)) {
    const pathTool = $canvasManager.get()?.tool.tools.path;
    const session = pathTool?.$editSession.get();
    const target = action.payload.entityIdentifier;
    if (
      pathTool &&
      session &&
      (target?.id !== session.entityIdentifier.id || target?.type !== session.entityIdentifier.type)
    ) {
      pathTool.requestEditExit(() => next(action));
      return action;
    }
  }
  return next(action);
};
