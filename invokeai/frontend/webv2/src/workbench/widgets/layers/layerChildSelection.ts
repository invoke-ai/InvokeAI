import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';

import { getLayerChildItem } from './layerChildRows';

/**
 * The panel's sub-selection: the one projected child row being edited. Widget
 * UI state, never document state — it creates no history, and it survives only
 * while its owner stays the selected layer and the item still exists.
 */
export interface LayerChildSelection {
  readonly projectId: string;
  readonly layerId: string;
  readonly itemId: string;
}

const store = createExternalStore<{ selection: LayerChildSelection | null }>({ selection: null });

export const selectLayerChild = (projectId: string, layerId: string, itemId: string): void =>
  store.setSnapshot({ selection: { itemId, layerId, projectId } });

export const clearLayerChildSelection = (): void => {
  if (store.getSnapshot().selection !== null) {
    store.setSnapshot({ selection: null });
  }
};

export const useLayerChildSelection = (): LayerChildSelection | null =>
  store.useSelector((snapshot) => snapshot.selection, Object.is);

export const getLayerChildSelection = (): LayerChildSelection | null => store.getSnapshot().selection;

/** Clears a selection whose owner is no longer the selected layer or whose item is gone. */
export const reconcileLayerChildSelection = (projectId: string, document: CanvasDocumentContractV3): void => {
  const selection = store.getSnapshot().selection;
  if (!selection || selection.projectId !== projectId) {
    return;
  }
  if (
    document.selectedLayerId !== selection.layerId ||
    !getLayerChildItem(document, selection.layerId, selection.itemId)
  ) {
    clearLayerChildSelection();
  }
};

registerAccountOwnedResource({ clear: clearLayerChildSelection, name: 'layer-child-selection' });
