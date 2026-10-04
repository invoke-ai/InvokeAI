import type { CanvasNodeContract } from '@workbench/canvas-engine/api';

import { useModelsSelector } from '@features/models';
import { getDocumentIndex, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { useCanvasEngine } from '@workbench/canvas-operations/react';
import { setLayerGroupExpanded } from '@workbench/layerPanelState';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { nextLayerName } from '@workbench/workbenchState';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { AddLayerItemId } from './addLayerMenu';

import { isAddLayerItemAvailable } from './addLayerMenu';
import { resolveDefaultControlModelForBase } from './controlModelOptions';
import {
  createControlLayer,
  createEmptyPaintLayer,
  createInpaintMaskLayer,
  createLayerId,
  createRegionalGuidanceLayer,
  createRegionalGuidanceLayerWithRefImage,
  nextControlLayerName,
  nextGroupName,
  nextInpaintMaskName,
  nextRegionalGuidanceName,
} from './layerOps';
import { useSelectedModelBase } from './useSelectedModelBase';

/**
 * Share guarded undoable creation across add menus and stack buttons. Insert above the selection, inside its group
 * for selected leaves.
 */
export const useAddLayer = (): ((id: AddLayerItemId) => void) => {
  const { t } = useTranslation();
  const engine = useCanvasEngine();
  const projectId = useActiveProjectId();
  const commitPrepared = usePreparedCommit(engine);
  const base = useSelectedModelBase();
  const models = useModelsSelector((snapshot) => snapshot.models);
  const layerNames = useActiveProjectSelector(
    (project) => getDocumentIndex(project.canvas.document).nodes.map((entry) => entry.node.name),
    (left, right) => left.length === right.length && left.every((name, index) => name === right[index])
  );
  const regionalGuidanceCount = useActiveProjectSelector(
    (project) => getDocumentLeaves(project.canvas.document).filter((layer) => layer.type === 'regional_guidance').length
  );

  return useCallback(
    (id: AddLayerItemId) => {
      if (!isAddLayerItemAvailable(id, base)) {
        return;
      }
      const add = (label: string, node: CanvasNodeContract): void => {
        commitPrepared(label, (model) =>
          model.prepare({ aboveId: model.document.selectedLayerId, nodes: [node], type: 'insert' })
        );
      };
      switch (id) {
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
          const layer = createEmptyPaintLayer(nextLayerName(layerNames));
          add(t('widgets.layers.actions.addRasterLayer'), layer);
          return;
        }
        case 'control': {
          const layer = createControlLayer(
            nextControlLayerName(layerNames),
            undefined,
            base,
            resolveDefaultControlModelForBase(models, base)
          );
          add(t('widgets.layers.actions.addControlLayer'), layer);
          return;
        }
        case 'inpaint_mask': {
          const layer = createInpaintMaskLayer(nextInpaintMaskName(layerNames));
          add(t('widgets.layers.actions.addInpaintMask'), layer);
          return;
        }
        case 'regional_guidance': {
          const layer = createRegionalGuidanceLayer(nextRegionalGuidanceName(layerNames), regionalGuidanceCount);
          add(t('widgets.layers.actions.addRegionalGuidance'), layer);
          return;
        }
        case 'regional_reference_image': {
          const layer = createRegionalGuidanceLayerWithRefImage(
            nextRegionalGuidanceName(layerNames),
            regionalGuidanceCount,
            base
          );
          add(t('widgets.layers.actions.addRegionalReferenceImage'), layer);
          return;
        }
      }
    },
    [base, commitPrepared, layerNames, models, projectId, regionalGuidanceCount, t]
  );
};
