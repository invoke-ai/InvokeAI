import { installCustomNodePack } from '@features/nodes/data/api';
import {
  addCustomNodeInstallLogEntry,
  removeCustomNodeInstallLogEntry,
  updateCustomNodeInstallLogEntry,
  useIsCustomNodeInstallRunning,
} from '@features/nodes/data/installLogStore';
import { refreshCustomNodePacks } from '@features/nodes/data/nodesStore';
import { useNotify } from '@features/nodes/ui/useNodesNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent } from '@platform/state/accountLifecycle';
import { getApiErrorMessage } from '@platform/transport/http';
import { useTranslation } from 'react-i18next';

export interface NodePackInstallOptions {
  /** A failed queue entry this install retries; it gives way to the new entry. */
  retryOf?: number;
  onInstalled?: () => void;
}

/**
 * Install a pack from a Git URL, tracking it in the install queue. One install runs at a time across the manager,
 * so `isInstalling` also covers installs other callers started.
 */
export const useNodePackInstall = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const isInstallRunning = useIsCustomNodeInstallRunning();
  const { isBusy, run } = useScopedAction();
  const isInstalling = isBusy || isInstallRunning;

  const install = (source: string, { onInstalled, retryOf }: NodePackInstallOptions = {}) => {
    if (isInstalling) {
      return;
    }

    let logEntryId: number | null = null;

    void run(
      async (owner) => {
        if (retryOf !== undefined) {
          removeCustomNodeInstallLogEntry(retryOf);
        }

        // Resolved in place below so the queue row settles with the install.
        const logEntry = addCustomNodeInstallLogEntry({ name: source, source, status: 'installing' });

        logEntryId = logEntry.id;

        const result = await installCustomNodePack(source, owner.signal);

        assertAccountScopeCurrent(owner);

        if (!result.success) {
          updateCustomNodeInstallLogEntry(logEntry.id, { message: result.message, name: result.name, status: 'error' });
          notify.error(t('nodes.installFailedTitle'), result.message);

          return;
        }

        updateCustomNodeInstallLogEntry(logEntry.id, {
          message: result.message,
          name: result.name,
          status: 'completed',
        });
        notify.success(
          t('nodes.installComplete'),
          result.workflows_imported > 0
            ? t('nodes.installCompleteWithWorkflows', { count: result.workflows_imported, name: result.name })
            : result.name
        );
        onInstalled?.();
        await refreshCustomNodePacks(owner);
        assertAccountScopeCurrent(owner);

        if (result.requires_dependencies) {
          // Sticky: this demands a manual pip install + restart; a toast that
          // expires quietly buries the instruction.
          notify.warning(
            t('nodes.dependenciesRequired'),
            t('nodes.dependenciesRequiredDescription', {
              dependencyFile: result.dependency_file ?? 'requirements.txt',
              name: result.name,
            }),
            { sticky: true }
          );
        }
      },
      (_message, error) => {
        const message = getApiErrorMessage(error, t('nodes.installFailed'));

        if (logEntryId !== null) {
          updateCustomNodeInstallLogEntry(logEntryId, { message, status: 'error' });
        }

        notify.error(t('nodes.installFailedTitle'), message);
      }
    );
  };

  return { install, isInstalling };
};
