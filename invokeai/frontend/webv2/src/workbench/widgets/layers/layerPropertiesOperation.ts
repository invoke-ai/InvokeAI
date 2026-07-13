import type { StartFilterOperationResult } from '@workbench/canvas-engine/engine';

interface LayerPropertiesOpenOwnership {
  requestToken: number | null;
  triggerOpen: boolean;
}

export type LayerFilterLaunchRejectedResult = Exclude<StartFilterOperationResult, 'started'>;
export type LayerFilterLaunchDisabledReason = LayerFilterLaunchRejectedResult | 'empty';

export interface LayerFilterLaunchEligibilityInput {
  hasEngine: boolean;
  hasExportableContent: boolean;
  isEnabled: boolean;
  isLocked: boolean;
}

const LAYER_FILTER_LAUNCH_REASON_KEYS: Record<LayerFilterLaunchDisabledReason, string> = {
  disabled: 'widgets.layers.actions.disabled',
  empty: 'widgets.layers.actions.empty',
  locked: 'widgets.layers.actions.locked',
  missing: 'widgets.layers.actions.missing',
  'not-ready': 'widgets.layers.actions.notReady',
  unsupported: 'widgets.layers.actions.unsupported',
};

export const getLayerFilterLaunchDisabledReason = ({
  hasEngine,
  hasExportableContent,
  isEnabled,
  isLocked,
}: LayerFilterLaunchEligibilityInput): LayerFilterLaunchDisabledReason | null => {
  if (!hasEngine) {
    return 'not-ready';
  }
  if (!isEnabled) {
    return 'disabled';
  }
  if (isLocked) {
    return 'locked';
  }
  if (!hasExportableContent) {
    return 'empty';
  }
  return null;
};

export const getLayerFilterLaunchReasonKey = (reason: LayerFilterLaunchDisabledReason): string =>
  LAYER_FILTER_LAUNCH_REASON_KEYS[reason];

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
  onOperationStarted: () => void,
  onOperationRejected?: (result: LayerFilterLaunchRejectedResult) => void
): StartFilterOperationResult | undefined => {
  const result = launch();
  if (result === 'started') {
    onOperationStarted();
  } else if (result !== undefined) {
    onOperationRejected?.(result);
  }
  return result;
};
