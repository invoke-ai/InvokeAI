import type { CanvasDocumentContractV3, LayerStackKind } from '@workbench/canvas-engine/api';
import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';
import type { LucideIcon } from 'lucide-react';

import { toaster } from '@platform/ui';
import { canMergeVisibleRasters, compileDocumentNodes, getDocumentLeaves } from '@workbench/canvas-engine/api';
import { useCanvasEngineRead } from '@workbench/widgets/canvas/engineStoreHooks';
import { usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useActiveProjectName } from '@workbench/WorkbenchContext';
import { EyeIcon, EyeOffIcon, FileDownIcon, LayersIcon, PlusIcon } from 'lucide-react';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { stackAddItemId } from './addLayerMenu';
import {
  canExportRasterPsd,
  getStackActions,
  isStackAllVisible,
  planStackVisibilityToggle,
  stackVisibilityAxis,
  type StackActionId,
} from './layerStackActions';
import { getPsdExportNoticeKey } from './psdExportNotice';
import { useAddLayer } from './useAddLayer';

export type LayerStackActionsEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'layers'>;

/** One stack action as the header buttons and the stack menu both render it. */
export interface LayerStackAction {
  readonly id: StackActionId;
  readonly label: string;
  readonly icon: LucideIcon;
  readonly disabled: boolean;
  run(): void;
}

/** The stack's actions with their enablement and handlers, shared by every surface that offers them. */
export const useLayerStackActions = (
  stack: LayerStackKind,
  document: CanvasDocumentContractV3,
  engine: LayerStackActionsEngine | null,
  editingLocked: boolean
): readonly LayerStackAction[] => {
  const commitPrepared = usePreparedCommit(engine);
  const { t } = useTranslation();
  const addLayer = useAddLayer();
  const projectName = useActiveProjectName();
  const axis = stackVisibilityAxis(stack);
  const { stacks } = document;
  // Read from the forests, so selection and other document-level changes cost nothing here.
  const { allVisible, exportable, nodes } = useMemo(() => {
    const own = compileDocumentNodes({ stacks }).filter((node) => node.stack === stack);
    return {
      allVisible: isStackAllVisible(own, axis),
      exportable: stack === 'raster' && canExportRasterPsd(getDocumentLeaves({ stacks })),
      nodes: own,
    };
  }, [axis, stack, stacks]);
  const canMerge = useCanvasEngineRead(
    engine,
    () =>
      !editingLocked &&
      !!engine &&
      stack === 'raster' &&
      canMergeVisibleRasters(engine.document.model()?.compileLeaves() ?? [], engine.exports.hasExportableLayerContent)
  );
  const canExport = !!engine && exportable;

  return useMemo(() => {
    const actions: Record<StackActionId, LayerStackAction> = {
      exportPsd: {
        disabled: editingLocked || !canExport,
        icon: FileDownIcon,
        id: 'exportPsd',
        label: t('widgets.layers.groupActions.exportPsd'),
        run: () => {
          if (!engine) {
            return;
          }
          void engine.exports
            .exportRasterLayersToPsd(projectName)
            .then((result) => {
              const noticeKey = getPsdExportNoticeKey(result);
              if (noticeKey) {
                toaster.create({ title: t(noticeKey), type: 'warning' });
              }
            })
            .catch(() => toaster.create({ title: t('widgets.layers.groupActions.exportFailed'), type: 'error' }));
        },
      },
      mergeVisible: {
        disabled: !canMerge,
        icon: LayersIcon,
        id: 'mergeVisible',
        label: t('widgets.layers.groupActions.mergeVisible'),
        run: () => {
          if (!engine) {
            return;
          }
          void engine.layers.mergeVisibleRasterLayers().then((result) => {
            if (result === 'not-ready') {
              toaster.create({ title: t('widgets.layers.groupActions.mergeNotReady'), type: 'warning' });
            } else if (result === 'over-budget') {
              toaster.create({ title: t('widgets.layers.groupActions.mergeOverBudget'), type: 'warning' });
            } else if (result === 'failed') {
              toaster.create({ title: t('widgets.layers.actions.operationFailed'), type: 'error' });
            }
          });
        },
      },
      new: {
        disabled: editingLocked,
        icon: PlusIcon,
        id: 'new',
        label: t('widgets.layers.groupActions.new'),
        run: () => addLayer(stackAddItemId(stack)),
      },
      toggleVisibility: {
        disabled: editingLocked,
        icon: allVisible ? EyeIcon : EyeOffIcon,
        id: 'toggleVisibility',
        label: t(allVisible ? 'widgets.layers.groupActions.hideAll' : 'widgets.layers.groupActions.showAll'),
        run: () => {
          const { ids, nextVisible } = planStackVisibilityToggle(nodes, axis);
          if (ids.length === 0) {
            return;
          }
          if (axis === 'hidden') {
            commitPrepared(t('widgets.layers.groupActions.toggleHidden'), (model) =>
              model.prepare({ type: 'set-hidden', updates: ids.map((id) => ({ id, isHidden: !nextVisible })) })
            );
            return;
          }
          commitPrepared(t('widgets.layers.groupActions.toggleVisibility'), (model) =>
            model.prepare({ type: 'set-enabled', updates: ids.map((id) => ({ id, isEnabled: nextVisible })) })
          );
        },
      },
    };
    return getStackActions(stack).map((id) => actions[id]);
  }, [
    addLayer,
    allVisible,
    axis,
    canExport,
    canMerge,
    commitPrepared,
    editingLocked,
    engine,
    nodes,
    projectName,
    stack,
    t,
  ]);
};
