import type { StartFilterOperationResult } from '@workbench/canvas-engine/engine';

interface LayerPropertiesOpenOwnership {
  requestToken: number | null;
  triggerOpen: boolean;
}

export const getLayerPropertiesOwnershipKey = (editingLocked: boolean): 'editable' | 'locked' =>
  editingLocked ? 'locked' : 'editable';

export const isLayerPropertiesOpen = ({ requestToken, triggerOpen }: LayerPropertiesOpenOwnership): boolean =>
  triggerOpen || requestToken !== null;

export const closeLayerPropertiesForOperation = ({ requestToken }: LayerPropertiesOpenOwnership) => ({
  requestToken: null,
  requestTokenToClear: requestToken,
  triggerOpen: false,
});

export const runLayerFilterOperation = (
  launch: () => StartFilterOperationResult | undefined,
  onOperationStarted: () => void
): StartFilterOperationResult | undefined => {
  const result = launch();
  if (result === 'started') {
    onOperationStarted();
  }
  return result;
};
