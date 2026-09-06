import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';

export const getIsCanvasMergeDownSupported = (
  selectedEntityIdentifier: CanvasEntityIdentifier | null,
  entityIdentifierBelowThisOne: CanvasEntityIdentifier | null
): boolean => {
  if (!selectedEntityIdentifier || !entityIdentifierBelowThisOne) {
    return false;
  }

  if (selectedEntityIdentifier.type === 'vector_layer' || entityIdentifierBelowThisOne.type === 'vector_layer') {
    return selectedEntityIdentifier.type === 'vector_layer' && entityIdentifierBelowThisOne.type === 'vector_layer';
  }

  return true;
};

export const getIsCanvasMergeDownHotkeyEnabled = (
  selectedEntityIdentifier: CanvasEntityIdentifier | null,
  entityIdentifierBelowThisOne: CanvasEntityIdentifier | null,
  isBusy: boolean
): boolean => {
  if (!getIsCanvasMergeDownSupported(selectedEntityIdentifier, entityIdentifierBelowThisOne)) {
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
  if (selectedEntityIdentifier.type === 'vector_layer') {
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
