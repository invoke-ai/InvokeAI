import type { LayoutPreset } from '@workbench/layoutContracts';

import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { closeLayoutPresetAdmin, layoutPresetManagerStore } from './layoutPresetManagerStore';

const EMPTY_LAYOUT_PRESETS: LayoutPreset[] = [];

/** Hosts the shared delete confirmation once; triggers only select the workspace id. */
export const LayoutPresetAdminDialogs = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const customPresets = useWorkbenchSelector(
    (snapshot) => snapshot.account.customLayoutPresets ?? EMPTY_LAYOUT_PRESETS
  );
  const deletePresetId = layoutPresetManagerStore.useSelector((snapshot) => snapshot.deletePresetId);
  const deleteTarget = customPresets.find((preset) => preset.id === deletePresetId) ?? null;
  const confirmDelete = useCallback(() => {
    if (deleteTarget) {
      layout.deletePreset(deleteTarget.id);
    }
  }, [deleteTarget, layout]);

  return (
    <ConfirmDialog
      body={t('topbar.presets.deleteBody', { name: deleteTarget?.label ?? t('topbar.presets.workspace') })}
      confirmLabel={t('topbar.presets.delete')}
      isOpen={deleteTarget !== null}
      title={t('topbar.presets.deleteQuestion')}
      onClose={closeLayoutPresetAdmin}
      onConfirm={confirmDelete}
    />
  );
};
