import type { LayoutPreset } from '@workbench/layoutContracts';

import { useExitRetainedValue } from '@platform/react/useExitRetainedValue';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { getLayoutPresetSourceOptions } from '@workbench/layoutPresetRouting';
import { getLayoutPresetLabel, layoutPresets } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import type { LayoutPresetDialogValue } from './layoutPresetDialogModel';

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { closeLayoutPresetAdmin, layoutPresetManagerStore } from './layoutPresetManagerStore';

const EMPTY_LAYOUT_PRESETS: LayoutPreset[] = [];

/** Host shared preset dialogs once; triggers only select the preset id. */
export const LayoutPresetAdminDialogs = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const customPresets = account.customLayoutPresets ?? EMPTY_LAYOUT_PRESETS;
  const { deletePresetId, editPresetId } = layoutPresetManagerStore.useSelector((snapshot) => snapshot);
  const editDialog = useExitRetainedValue(editPresetId);
  const editTarget = useMemo(() => {
    if (!editDialog.value) {
      return null;
    }

    const presetId = editDialog.value;
    const exists = [...layoutPresets, ...customPresets].some((preset) => preset.id === presetId);

    return exists ? resolveSavedLayoutPreset(account, presetId) : null;
  }, [account, customPresets, editDialog.value]);
  const deleteTarget = customPresets.find((preset) => preset.id === deletePresetId) ?? null;
  const sourceOptions = useMemo(() => (editTarget ? getLayoutPresetSourceOptions(editTarget) : []), [editTarget]);

  const submitEdit = useCallback(
    ({ defaultRoute, iconId, name }: LayoutPresetDialogValue) => {
      if (editTarget) {
        const shipped = layoutPresets.find((preset) => preset.id === editTarget.id);

        layout.setPresetRoute(editTarget.id, defaultRoute);
        layout.renamePreset(editTarget.id, name, shipped ? getLayoutPresetLabel(shipped, t) : undefined);
        layout.setPresetIcon(editTarget.id, iconId);
      }
    },
    [editTarget, layout, t]
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
          key={editDialog.generation}
          defaultRoute={editTarget.defaultRoute}
          iconId={editTarget.iconId}
          isOpen={editDialog.isOpen}
          name={getLayoutPresetLabel(editTarget, t)}
          sourceOptions={sourceOptions}
          submitLabel={t('topbar.presets.save')}
          title={t('topbar.presets.edit')}
          onClose={closeLayoutPresetAdmin}
          onExitComplete={editDialog.release}
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
