import type { NodePackInfo } from '@features/nodes/core/catalog';

import { uninstallCustomNodePack } from '@features/nodes/data/api';
import { addCustomNodeInstallLogEntry } from '@features/nodes/data/installLogStore';
import { refreshCustomNodePacks, removeCustomNodePackFromStore } from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useTranslation } from 'react-i18next';

export const useNodePackActions = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { isBusy: isUninstalling, run } = useScopedAction();

  const uninstall = (pack: NodePackInfo, onUninstalled?: (packName: string) => void) =>
    run(
      async (owner) => {
        const result = await uninstallCustomNodePack(pack.name, owner.signal);

        assertAccountScopeCurrent(owner);
        removeCustomNodePackFromStore(pack.name);
        addCustomNodeInstallLogEntry({ message: result.message, name: result.name, status: 'uninstalled' });
        notify.success(t('nodes.uninstalledTitle'), t('nodes.uninstalledDescription'));
        onUninstalled?.(pack.name);
      },
      (_message, error) => {
        notify.error(t('nodes.uninstallFailed'), getApiErrorMessage(error, t('nodes.couldNotUninstall')));
        // The scope is current when onError runs; resync with the backend.
        void refreshCustomNodePacks();
      }
    );

  return { isUninstalling, uninstall };
};
