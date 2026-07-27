import type { NodePackInfo } from '@features/nodes/core/catalog';

import { uninstallCustomNodePack } from '@features/nodes/data/api';
import { addCustomNodeInstallLogEntry } from '@features/nodes/data/installLogStore';
import { refreshCustomNodePacks, removeCustomNodePackFromStore } from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';

export const useNodePackActions = () => {
  const notify = useNotify();

  const uninstall = async (pack: NodePackInfo, onUninstalled?: (packName: string) => void) => {
    const owner = captureAccountScope();

    try {
      const result = await uninstallCustomNodePack(pack.name, owner.signal);

      assertAccountScopeCurrent(owner);
      removeCustomNodePackFromStore(pack.name);
      addCustomNodeInstallLogEntry({ message: result.message, name: result.name, status: 'uninstalled' });
      notify.success('Node pack uninstalled', 'Restart InvokeAI for node removal to take full effect.');
      onUninstalled?.(pack.name);
    } catch (error) {
      if (!isAccountScopeCurrent(owner)) {
        return;
      }

      notify.error('Uninstall failed', getApiErrorMessage(error, 'Could not uninstall the node pack.'));
      await refreshCustomNodePacks(owner);
    }
  };

  return { uninstall };
};
