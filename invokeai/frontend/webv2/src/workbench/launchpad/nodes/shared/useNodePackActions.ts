import type { NodePackInfo } from '@workbench/customNodes/api';

import { getApiErrorMessage } from '@workbench/backend/http';
import { uninstallCustomNodePack } from '@workbench/customNodes/api';
import { addCustomNodeInstallLogEntry } from '@workbench/customNodes/installLogStore';
import { refreshCustomNodePacks, removeCustomNodePackFromStore } from '@workbench/customNodes/nodesStore';
import { useNotify } from '@workbench/useNotify';

export const useNodePackActions = () => {
  const notify = useNotify();

  const uninstall = async (pack: NodePackInfo, onUninstalled?: (packName: string) => void) => {
    try {
      const result = await uninstallCustomNodePack(pack.name);

      removeCustomNodePackFromStore(pack.name);
      addCustomNodeInstallLogEntry({ message: result.message, name: result.name, status: 'uninstalled' });
      notify.success('Node pack uninstalled', 'Restart InvokeAI for node removal to take full effect.');
      onUninstalled?.(pack.name);
    } catch (error) {
      notify.error('Uninstall failed', getApiErrorMessage(error, 'Could not uninstall the node pack.'));
      await refreshCustomNodePacks();
    }
  };

  return { uninstall };
};
