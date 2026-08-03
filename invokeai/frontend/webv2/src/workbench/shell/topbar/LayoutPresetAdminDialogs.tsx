import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { closeLayoutPresetAdmin, layoutPresetManagerStore } from './layoutPresetManagerStore';

/**
 * Hosts the edit and delete dialogs shared by every preset-management surface.
 * The trigger only writes a preset id to the store, which avoids mounting a
 * second copy of these dialogs in both the strip and the manager.
 */
export const LayoutPresetAdminDialogs = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const customPresets = useWorkbenchSelector((snapshot) => snapshot.account.customLayoutPresets ?? []);
  const { deletePresetId, editPresetId } = layoutPresetManagerStore.useSelector((snapshot) => snapshot);
  const editTarget = customPresets.find((preset) => preset.id === editPresetId) ?? null;
  const deleteTarget = customPresets.find((preset) => preset.id === deletePresetId) ?? null;

  const submitEdit = useCallback(
    ({ iconId, name }: { iconId: string; name: string }) => {
      if (editTarget) {
        layout.renamePreset(editTarget.id, name);
        layout.setPresetIcon(editTarget.id, iconId);
      }
    },
    [editTarget, layout]
  );
  const confirmDelete = useCallback(() => {
    if (deleteTarget) {
      layout.deletePreset(deleteTarget.id);
    }
  }, [deleteTarget, layout]);

  return (
    <>
      {editTarget ? (
        <LayoutPresetDialog
          iconId={editTarget.iconId}
          isOpen
          name={editTarget.label}
          submitLabel={t('topbar.presets.save')}
          title={t('topbar.presets.edit')}
          onClose={closeLayoutPresetAdmin}
          onSubmit={submitEdit}
        />
      ) : null}
      <ConfirmDialog
        body={t('topbar.presets.deleteBody', { name: deleteTarget?.label ?? t('topbar.presets.layoutPreset') })}
        confirmLabel={t('topbar.presets.delete')}
        isOpen={deleteTarget !== null}
        title={t('topbar.presets.deleteQuestion')}
        onClose={closeLayoutPresetAdmin}
        onConfirm={confirmDelete}
      />
    </>
  );
};
