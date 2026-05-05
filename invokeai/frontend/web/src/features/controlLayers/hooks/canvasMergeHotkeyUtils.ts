import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';

export const getIsCanvasMergeDownHotkeyEnabled = (
  selectedEntityIdentifier: CanvasEntityIdentifier | null,
  entityIdentifierBelowThisOne: CanvasEntityIdentifier | null,
  isBusy: boolean
): boolean => {
  if (!selectedEntityIdentifier || !entityIdentifierBelowThisOne) {
    return false;
  }
  if (isBusy) {
    return false;
  }
  return true;
};

export const getIsCanvasMergeVisibleHotkeyEnabled = (
  selectedEntityIdentifier: CanvasEntityIdentifier | null,
  visibleEntityCount: number,
  isBusy: boolean
): boolean => {
  if (!selectedEntityIdentifier) {
    return false;
  }
  if (visibleEntityCount <= 1) {
    return false;
  }
  if (isBusy) {
    return false;
  }
  return true;
};
