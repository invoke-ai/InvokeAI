import type { CanvasDocumentContractV3 } from '@workbench/canvas-engine/api';
import type { CanvasProjectMutationDispatch } from '@workbench/useCanvasProjectMutationDispatch';
import type { PreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';

import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { publishLayerPanelSelection, setLayerChildrenCollapsed, setLayerPanelFocus } from '@workbench/layerPanelState';

import { getLayerChildItem, layerChildRowKey, MASK_GUIDANCE_ITEM_ID } from './layerChildRows';
import { DEFAULT_MASK_GUIDANCE } from './layerOps';

/** The one child row being edited: widget UI state, never document state or history. */
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

/**
 * Adds an inpaint mask's regional guidance as one undo step, then selects the mask alone and, under it, the new row.
 * False when nothing was added.
 */
export const addMaskGuidance = ({
  commit,
  dispatch,
  label,
  layerId,
  projectId,
}: {
  commit: PreparedCommit;
  /** Bound to `projectId`. */
  dispatch: CanvasProjectMutationDispatch;
  label: string;
  layerId: string;
  projectId: string;
}): boolean => {
  const outcome = commit(label, (model) => {
    const layer = model.getLayer(layerId);
    if (layer?.type !== 'inpaint_mask') {
      return { ids: [layerId], status: 'missing' };
    }
    // One that appeared since the caller rendered must not be stomped by the empty default.
    if (layer.guidance) {
      return { status: 'unchanged' };
    }
    return model.prepare({
      before: { guidance: null, layerType: 'inpaint_mask' },
      config: { guidance: { ...DEFAULT_MASK_GUIDANCE }, layerType: 'inpaint_mask' },
      id: layerId,
      type: 'patch-config',
    });
  });
  if (outcome.status !== 'committed') {
    return false;
  }
  // Published before the dispatch so panel reconciliation keeps it; the mask is selected in this tick because
  // reconciliation clears a row selection whose owner is not the selected layer.
  publishLayerPanelSelection({ primaryId: layerId, projectId, selectedIds: [layerId] });
  dispatch({ id: layerId, type: 'setCanvasSelectedLayer' });
  setLayerChildrenCollapsed(projectId, layerId, layerId, false);
  selectLayerChild(projectId, layerId, MASK_GUIDANCE_ITEM_ID);
  setLayerPanelFocus(projectId, layerId, layerChildRowKey(layerId, MASK_GUIDANCE_ITEM_ID));
  return true;
};

registerAccountOwnedResource({ clear: clearLayerChildSelection, name: 'layer-child-selection' });
