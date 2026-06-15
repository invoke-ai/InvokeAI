import { useCallback, useState } from 'react';

import { getAccessTokenForSource } from '@workbench/models/apiKeys';
import { installModel, type InstallModelRequest } from '@workbench/models/api';
import { addInstallJob } from '@workbench/models/installsStore';
import { useNotify } from '@workbench/useNotify';

/**
 * Shared install entry point: queues an install job, optimistically adds it to
 * the install store, and surfaces a notification. A saved Civitai API key is
 * attached automatically for civitai.com URLs unless the caller supplies an
 * explicit token. Bulk flows pass `silent` and emit one summary notice instead
 * of a toast per model; failures always notify.
 */
export const useInstallActions = () => {
  const notify = useNotify();
  const [pendingSources, setPendingSources] = useState<ReadonlySet<string>>(new Set());

  const install = useCallback(
    async (request: InstallModelRequest, options?: { silent?: boolean }): Promise<boolean> => {
      setPendingSources((current) => new Set(current).add(request.source));

      try {
        const job = await installModel({
          ...request,
          accessToken: request.accessToken ?? getAccessTokenForSource(request.source),
        });

        addInstallJob(job);

        if (!options?.silent) {
          notify.success('Model install queued', request.source);
        }

        return true;
      } catch (error) {
        notify.error('Model install failed to start', error instanceof Error ? error.message : String(error));

        return false;
      } finally {
        setPendingSources((current) => {
          const next = new Set(current);

          next.delete(request.source);

          return next;
        });
      }
    },
    [notify]
  );

  return { install, pendingSources };
};
