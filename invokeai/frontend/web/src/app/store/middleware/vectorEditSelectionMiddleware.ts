import type { Middleware } from '@reduxjs/toolkit';
import { isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import { canvasReset } from 'features/controlLayers/store/actions';
import {
  allEntitiesDeleted,
  canvasMetadataRecalled,
  canvasProjectRecalled,
  canvasSnapshotRestored,
  entityDeleted,
  entityReset,
  entitySelected,
} from 'features/controlLayers/store/canvasSlice';
import { $canvasManager } from 'features/controlLayers/store/ephemeral';
import { selectCanvasSlice } from 'features/controlLayers/store/selectors';

// Defer explicit layer selection before it reaches the reducer, preserving the edit session on Cancel.
export const vectorEditSelectionMiddleware: Middleware = (api) => (next) => (action) => {
  const pathTool = $canvasManager.get()?.tool.tools.path;
  const session = pathTool?.$editSession.get();
  if (
    session &&
    pathTool &&
    (isAnyOf(
      canvasReset,
      allEntitiesDeleted,
      canvasMetadataRecalled,
      canvasProjectRecalled,
      canvasSnapshotRestored
    )(action) ||
      (isAnyOf(entityDeleted, entityReset)(action) &&
        action.payload.entityIdentifier.id === session.entityIdentifier.id))
  ) {
    pathTool.acceptEditSession();
  }
  if (entitySelected.match(action)) {
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
  const result = next(action);
  const remainingSession = pathTool?.$editSession.get();
  if (remainingSession && pathTool) {
    const canvas = selectCanvasSlice(api.getState() as RootState);
    if (
      !canvas.vectorLayers.entities.some((layer) => layer.id === remainingSession.entityIdentifier.id) ||
      canvas.selectedEntityIdentifier?.id !== remainingSession.entityIdentifier.id ||
      canvas.selectedEntityIdentifier.type !== 'vector_layer'
    ) {
      pathTool.acceptEditSession();
    }
  }
  return result;
};
