import { installModel, type InstallModelRequest } from '@features/models/data/api';
import { getAccessTokenForSource } from '@features/models/data/apiKeys';
import { addInstallJob } from '@features/models/data/installsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import {
  assertAccountScopeCurrent,
  captureAccountScope,
  isAccountScopeCurrent,
} from '@platform/state/accountLifecycle';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * Shared install entry point: queues an install job, optimistically adds it to
 * the install store, and surfaces a notification. A saved Civitai API key is
 * attached automatically for civitai.com URLs unless the caller supplies an
 * explicit token. Bulk flows pass `silent` and emit one summary notice instead
 * of a toast per model; failures always notify.
 */
export const useInstallActions = () => {
  const { t } = useTranslation();
  const notify = useNotify();
  const [pendingSources, setPendingSources] = useState<ReadonlySet<string>>(new Set());

  const install = useCallback(
    async (request: InstallModelRequest, options?: { silent?: boolean }): Promise<boolean> => {
      const owner = captureAccountScope();

      setPendingSources((current) => new Set(current).add(request.source));

      try {
        const job = await installModel(
          {
            ...request,
            accessToken: request.accessToken ?? getAccessTokenForSource(request.source),
          },
          owner.signal
        );

        assertAccountScopeCurrent(owner);
        addInstallJob(job);

        if (!options?.silent) {
          notify.success(t('models.modelInstallQueued'), request.source);
        }

        return true;
      } catch (error) {
        if (!isAccountScopeCurrent(owner)) {
          return false;
        }

        notify.error(t('models.modelInstallFailedToStart'), error instanceof Error ? error.message : String(error));

        return false;
      } finally {
        if (isAccountScopeCurrent(owner)) {
          setPendingSources((current) => {
            const next = new Set(current);

            next.delete(request.source);

            return next;
          });
        }
      }
    },
    [notify, t]
  );

  return { install, pendingSources };
};
