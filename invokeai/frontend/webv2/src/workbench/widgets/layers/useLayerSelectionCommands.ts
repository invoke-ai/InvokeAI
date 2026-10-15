import type { CanvasEngineHandle } from '@workbench/canvas-operations/react';

import { toaster } from '@platform/ui';
import { publishLayerPanelSelection } from '@workbench/layerPanelState';
import { reportLayerOperation, usePreparedCommit } from '@workbench/widgets/canvas/useStructuralCommit';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { groupLayers } from './layerGroupCommands';

type SelectionEngine = Pick<CanvasEngineHandle, 'document' | 'exports' | 'interaction' | 'layers'>;

export interface LayerSelectionCommands {
  canDelete: boolean;
  canDuplicate: boolean;
  canGroup: boolean;
  deleteSelected: () => void;
  duplicateSelected: () => void;
  groupSelected: () => void;
}

/** Use model-authoritative eligibility for duplicate/group/delete on the current selection. */
export const useLayerSelectionCommands = (
  engine: SelectionEngine | null,
  projectId: string,
  selectedIds: readonly string[],
  editingLocked: boolean
): LayerSelectionCommands => {
  const commitPrepared = usePreparedCommit(engine);
  const { t } = useTranslation();
  const model = engine?.document.model() ?? null;
  const none = selectedIds.length === 0;
  const canGroup =
    !editingLocked &&
    !none &&
    !!model &&
    model.refusalFor({ groupId: '\0probe', ids: selectedIds, name: '', type: 'group' }) === null;
  const canDelete =
    !editingLocked && !none && !!model && model.refusalFor({ ids: selectedIds, type: 'remove' }) === null;
  const canDuplicate = !editingLocked && !none && !!engine;

  const duplicateSelected = useCallback(async (): Promise<void> => {
    try {
      const result = await engine?.layers.duplicateLayers(selectedIds);
      if (result?.status === 'duplicated') {
        publishLayerPanelSelection({
          primaryId: result.selectedLayerId,
          projectId,
          selectedIds: result.duplicateIds,
        });
        return;
      }
      if (result?.status === 'busy') {
        return;
      }
      if (result && result.status !== 'nothing') {
        reportLayerOperation(
          result.status,
          (title, description) => toaster.create({ description, title, type: 'warning' }),
          t
        );
        return;
      }
    } catch {
      // Reducer rejection is failure-atomic; surface the same actionable result as a refusal.
    }
    if (engine) {
      toaster.create({ title: t('widgets.layers.actions.copyFailed'), type: 'warning' });
    }
  }, [engine, projectId, selectedIds, t]);

  const groupSelected = useCallback(
    () => groupLayers(engine, projectId, selectedIds, t('widgets.layers.actions.groupSelected')),
    [engine, projectId, selectedIds, t]
  );

  const deleteSelected = useCallback(() => {
    commitPrepared(t('widgets.layers.actions.deleteSelected'), (model) =>
      model.prepare({ ids: selectedIds, type: 'remove' })
    );
  }, [commitPrepared, selectedIds, t]);

  const runDuplicate = useCallback(() => {
    void duplicateSelected();
  }, [duplicateSelected]);

  return useMemo(
    () => ({
      canDelete,
      canDuplicate,
      canGroup,
      deleteSelected,
      duplicateSelected: runDuplicate,
      groupSelected,
    }),
    [canDelete, canDuplicate, canGroup, deleteSelected, groupSelected, runDuplicate]
  );
};
