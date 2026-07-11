import type { CanvasEngine } from '@workbench/canvas-engine/engine';

import { useCanvasEngine } from '@workbench/widgets/canvas/useCanvasEngine';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { nextLayerName } from '@workbench/workbenchState';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { AddLayerItemId } from './addLayerMenu';

import {
  applyStructural,
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createRegionalGuidanceLayer,
  createRegionalGuidanceLayerWithRefImage,
  nextControlLayerName,
  nextInpaintMaskName,
  nextRegionalGuidanceName,
} from './layerOps';
import { useSelectedModelBase } from './useSelectedModelBase';

/**
 * Returns a single `addLayer(id)` callback that creates a new layer of the given
 * kind at the top of the stack via the shared `applyStructural` seam (one undoable
 * history entry per add). Reused by the panel's add-layer menu AND each group
 * header's "New" button so both surfaces stay in lockstep.
 */
export const useAddLayer = (): ((id: AddLayerItemId) => void) => {
  const { t } = useTranslation();
  const engine: CanvasEngine | null = useCanvasEngine();
  const dispatch = useWorkbenchDispatch();
  const base = useSelectedModelBase();
  const layerNames = useActiveProjectSelector(
    (project) => project.canvas.document.layers.map((layer) => layer.name),
    (left, right) => left.length === right.length && left.every((name, index) => name === right[index])
  );
  const regionalGuidanceCount = useActiveProjectSelector(
    (project) => project.canvas.document.layers.filter((layer) => layer.type === 'regional_guidance').length
  );

  return useCallback(
    (id: AddLayerItemId) => {
      switch (id) {
        case 'raster': {
          const layer = createEmptyPaintLayer(nextLayerName(layerNames));
          applyStructural(
            engine,
            dispatch,
            t('widgets.layers.actions.addRasterLayer'),
            { index: 0, layer, type: 'addCanvasLayer' },
            { ids: [layer.id], type: 'removeCanvasLayers' }
          );
          return;
        }
        case 'control': {
          const layer = createControlLayer(nextControlLayerName(layerNames), undefined, base);
          applyStructural(
            engine,
            dispatch,
            t('widgets.layers.actions.addControlLayer'),
            { index: 0, layer, type: 'addCanvasLayer' },
            { ids: [layer.id], type: 'removeCanvasLayers' }
          );
          return;
        }
        case 'inpaint_mask': {
          const layer = createInpaintMaskLayer(nextInpaintMaskName(layerNames));
          applyStructural(
            engine,
            dispatch,
            t('widgets.layers.actions.addInpaintMask'),
            { index: 0, layer, type: 'addCanvasLayer' },
            { ids: [layer.id], type: 'removeCanvasLayers' }
          );
          return;
        }
        case 'regional_guidance': {
          const layer = createRegionalGuidanceLayer(nextRegionalGuidanceName(layerNames), regionalGuidanceCount);
          applyStructural(
            engine,
            dispatch,
            t('widgets.layers.actions.addRegionalGuidance'),
            { index: 0, layer, type: 'addCanvasLayer' },
            { ids: [layer.id], type: 'removeCanvasLayers' }
          );
          return;
        }
        case 'regional_reference_image': {
          const layer = createRegionalGuidanceLayerWithRefImage(
            nextRegionalGuidanceName(layerNames),
            regionalGuidanceCount,
            base
          );
          applyStructural(
            engine,
            dispatch,
            t('widgets.layers.actions.addRegionalReferenceImage'),
            { index: 0, layer, type: 'addCanvasLayer' },
            { ids: [layer.id], type: 'removeCanvasLayers' }
          );
          return;
        }
      }
    },
    [base, dispatch, engine, layerNames, regionalGuidanceCount, t]
  );
};
