import type { LayoutPreset } from '@workbench/layoutContracts';

import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { getLayoutPresetSourceOptions } from '@workbench/layoutPresetRouting';
import { layoutPresets } from '@workbench/layoutPresets';
import { resolveSavedLayoutPreset } from '@workbench/layoutPresetSnapshots';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { LayoutPresetDialog } from './LayoutPresetDialog';
import { closeLayoutPresetAdmin, layoutPresetManagerStore } from './layoutPresetManagerStore';

const EMPTY_LAYOUT_PRESETS: LayoutPreset[] = [];

/**
 * Hosts the edit and delete dialogs shared by every preset-management surface.
 * The trigger only writes a preset id to the store, which avoids mounting a
 * second copy of these dialogs in both the strip and the manager.
 */
export const LayoutPresetAdminDialogs = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const account = useWorkbenchSelector((snapshot) => snapshot.account);
  const customPresets = account.customLayoutPresets ?? EMPTY_LAYOUT_PRESETS;
  const { deletePresetId, editPresetId } = layoutPresetManagerStore.useSelector((snapshot) => snapshot);
  const editTarget = useMemo(() => {
    if (!editPresetId) {
      return null;
    }

    const exists = [...layoutPresets, ...customPresets].some((preset) => preset.id === editPresetId);

    return exists ? resolveSavedLayoutPreset(account, editPresetId) : null;
  }, [account, customPresets, editPresetId]);
  const deleteTarget = customPresets.find((preset) => preset.id === deletePresetId) ?? null;
  const sourceOptions = useMemo(() => (editTarget ? getLayoutPresetSourceOptions(editTarget) : []), [editTarget]);

  const submitEdit = useCallback(
    ({
      defaultRoute,
      iconId,
      name,
    }: {
      defaultRoute: Parameters<typeof layout.setPresetRoute>[1];
      iconId: string;
      name: string;
    }) => {
      if (editTarget) {
        layout.setPresetRoute(editTarget.id, defaultRoute);

        if (!editTarget.isBuiltIn) {
          layout.renamePreset(editTarget.id, name);
          layout.setPresetIcon(editTarget.id, iconId);
        }
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
          key={editTarget.id}
          defaultRoute={editTarget.defaultRoute}
          iconId={editTarget.iconId}
          isBuiltIn={editTarget.isBuiltIn}
          isOpen
          name={editTarget.label}
          sourceOptions={sourceOptions}
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
