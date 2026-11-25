import type { CanvasNodeContract } from '@workbench/canvas-engine/api';

import {
  advanceMaskFillColors,
  createControlLayer,
  getDocumentIndex,
  nextControlLayerName,
  nextMaskFillColor,
} from '@workbench/canvas-engine/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { setLayerGroupExpanded } from '@workbench/layerPanelState';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { AddLayerItemId } from './addLayerMenu';

import { isAddLayerItemAvailable } from './addLayerMenu';
import { prepareNewRasterLayer } from './layerCommands';
import {
  createInpaintMaskLayer,
  createLayerId,
  createRegionalGuidanceLayer,
  createRegionalGuidanceLayerWithRefImage,
  nextGroupName,
  nextInpaintMaskName,
  nextRegionalGuidanceName,
} from './layerOps';
import { requestLayerProperties } from './layerPropertiesRequestStore';
import { useSelectedModelBase } from './useSelectedModelBase';

/** Guarded undoable creation, above the selection and inside its group for selected leaves. */
export const useAddLayer = (): ((id: AddLayerItemId) => void) => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const commitPrepared = usePreparedCommit(engine);
  const base = useSelectedModelBase();
  const layerNames = useActiveProjectSelector(
    (project) => getDocumentIndex(project.canvas.document).nodes.map((entry) => entry.node.name),
    (left, right) => left.length === right.length && left.every((name, index) => name === right[index])
  );
  return useCallback(
    (id: AddLayerItemId) => {
      if (!isAddLayerItemAvailable(id, base)) {
        return;
      }
      const add = (label: string, node: CanvasNodeContract) => {
        const outcome = commitPrepared(label, (model) =>
          model.prepare({ aboveId: model.document.selectedLayerId, nodes: [node], type: 'insert' })
        );
        if (outcome.status === 'committed' && (node.type === 'inpaint_mask' || node.type === 'regional_guidance')) {
          advanceMaskFillColors(node.type);
        }
        return outcome;
      };
      switch (id) {
        case 'vector': {
          if (!engine) {
            return;
          }
          const layer = engine.layers.vectorEdit.createLayer({ id: createLayerId() });
          if (layer) {
            add(t('widgets.layers.actions.addVectorLayer'), layer);
          }
          return;
        }
        case 'group': {
          // An empty group has no stack of its own: it joins the selected node's stack, else raster.
          const groupId = createLayerId();
          const outcome = commitPrepared(t('widgets.layers.actions.newGroup'), (model) => {
            const selectedId = model.document.selectedLayerId;
            const selected = selectedId ? getDocumentIndex(model.document).byId.get(selectedId) : undefined;
            return model.prepare({
              aboveId: selectedId,
              nodes: [
                {
                  children: [],
                  id: groupId,
                  isEnabled: true,
                  isLocked: false,
                  name: nextGroupName(layerNames),
                  type: 'group',
                },
              ],
              stack: selected?.stack ?? 'raster',
              type: 'insert',
            });
          });
          if (outcome.status === 'committed') {
            setLayerGroupExpanded(projectId, groupId, [groupId], true);
          }
          return;
        }
        case 'raster': {
          commitPrepared(t('widgets.layers.actions.addRasterLayer'), prepareNewRasterLayer);
          return;
        }
        case 'control': {
          const layer = createControlLayer({ base, id: createLayerId(), name: nextControlLayerName(layerNames) });
          // A new control layer starts untyped: its Properties ask what it should control.
          if (add(t('widgets.layers.actions.addControlLayer'), layer).status === 'committed') {
            requestLayerProperties(layer.id);
          }
          return;
        }
        case 'inpaint_mask': {
          const layer = createInpaintMaskLayer(
            nextInpaintMaskName(layerNames),
            undefined,
            nextMaskFillColor('inpaint_mask')
          );
          add(t('widgets.layers.actions.addInpaintMask'), layer);
          return;
        }
        case 'regional_guidance': {
          const layer = createRegionalGuidanceLayer(
            nextRegionalGuidanceName(layerNames),
            nextMaskFillColor('regional_guidance')
          );
          add(t('widgets.layers.actions.addRegionalGuidance'), layer);
          return;
        }
        case 'regional_reference_image': {
          const layer = createRegionalGuidanceLayerWithRefImage(
            nextRegionalGuidanceName(layerNames),
            nextMaskFillColor('regional_guidance'),
            base
          );
          add(t('widgets.layers.actions.addRegionalReferenceImage'), layer);
          return;
        }
      }
    },
    [engine, base, commitPrepared, layerNames, projectId, t]
  );
};
