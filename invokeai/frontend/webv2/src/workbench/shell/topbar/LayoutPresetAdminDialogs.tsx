import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { getOrderedLayoutPresets } from '@workbench/layoutPresetCollection';
import { getLayoutPresetLabel } from '@workbench/layoutPresets';
import { useWorkbenchCommands, useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { closeLayoutPresetAdmin, layoutPresetManagerStore } from './layoutPresetManagerStore';

/** Hosts the shared delete and reset confirmations once; triggers only select the workspace and the action. */
export const LayoutPresetAdminDialogs = () => {
  const { t } = useTranslation();
  const { layout } = useWorkbenchCommands();
  const confirm = layoutPresetManagerStore.useSelector((snapshot) => snapshot.confirm);
  // The confirmation's workspace by its shown name, while it is still one the action applies to.
  const name = useWorkbenchSelector((snapshot) => {
    const preset =
      confirm &&
      getOrderedLayoutPresets(snapshot.account).find(
        ({ id, isBuiltIn }) => id === confirm.presetId && (confirm.action === 'restore') === (isBuiltIn === true)
      );

    return preset ? getLayoutPresetLabel(preset, t) : null;
  }, Object.is);
  const isRestore = confirm?.action === 'restore';
  const confirmAction = useCallback(() => {
    if (!confirm || name === null) {
      return;
    }
    if (confirm.action === 'restore') {
      layout.restorePresetDefault(confirm.presetId);
    } else {
      layout.deletePreset(confirm.presetId);
    }
  }, [confirm, layout, name]);

  return (
    <ConfirmDialog
      body={t(isRestore ? 'topbar.presets.restoreBody' : 'topbar.presets.deleteBody', {
        name: name ?? t('topbar.presets.workspace'),
      })}
      confirmLabel={t(isRestore ? 'topbar.presets.resetToDefault' : 'topbar.presets.delete')}
      isOpen={name !== null}
      title={t(isRestore ? 'topbar.presets.restoreQuestion' : 'topbar.presets.deleteQuestion')}
      onClose={closeLayoutPresetAdmin}
      onConfirm={confirmAction}
    />
  );
};
