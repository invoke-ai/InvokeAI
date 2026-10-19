import type { ModelInstallJob } from '@features/models/core/types';

import {
  cancelModelInstall,
  installModel,
  pauseModelInstall,
  restartFailedModelInstall,
  restartModelInstallFile,
  resumeModelInstall,
} from '@features/models/data/api';
import { getAccessTokenForSource } from '@features/models/data/apiKeys';
import {
  addInstallJob,
  dismissInstallJob,
  refreshInstalls,
  replaceInstallJob,
} from '@features/models/data/installsStore';
import { useNotify } from '@features/models/ui/useModelsNotify';
import { useScopedAction } from '@platform/react/useScopedAction';
import { assertAccountScopeCurrent, type AccountScope } from '@platform/state/accountLifecycle';
import { useTranslation } from 'react-i18next';

import { canRestartFailedParts, getInstallJobAccessToken, getInstallJobSourceString } from './queueModel';

const isInstallJob = (value: unknown): value is ModelInstallJob =>
  typeof value === 'object' && value !== null && 'id' in value;

/** Row actions share one busy flag so a second click cannot race the first request. */
export const useInstallJobActions = (job: ModelInstallJob) => {
  const { t } = useTranslation();
  const notify = useNotify();
  const { isBusy, run } = useScopedAction();

  const runAction = (action: (owner: AccountScope) => Promise<unknown>, failureTitle: string) =>
    run(
      async (owner) => {
        const result = await action(owner);

        assertAccountScopeCurrent(owner);

        if (isInstallJob(result)) {
          replaceInstallJob(result);
          // A coalesced refresh already in flight can clobber the optimistic row; revalidate to converge.
          void refreshInstalls(owner);
        } else {
          await refreshInstalls(owner);
        }
      },
      (message) => notify.error(failureTitle, message)
    );

  const resubmit = async (owner: AccountScope) => {
    const source = getInstallJobSourceString(job);
    const created = await installModel(
      {
        accessToken: getInstallJobAccessToken(job) ?? getAccessTokenForSource(source),
        config: job.config_in ?? undefined,
        inplace: job.inplace,
        source,
      },
      owner.signal
    );

    assertAccountScopeCurrent(owner);
    dismissInstallJob(job.id);
    addInstallJob(created);
  };

  return {
    cancel: () => void runAction((owner) => cancelModelInstall(job.id, owner.signal), t('models.cancelFailed')),
    dismiss: () => dismissInstallJob(job.id),
    isBusy,
    pause: () => void runAction((owner) => pauseModelInstall(job.id, owner.signal), t('models.pauseFailed')),
    restartFile: (fileUrl: string) =>
      void runAction((owner) => restartModelInstallFile(job.id, fileUrl, owner.signal), t('models.restartFileFailed')),
    resume: () => void runAction((owner) => resumeModelInstall(job.id, owner.signal), t('models.resumeFailed')),
    // The backend restarts only failed parts of a live job; anything else is queued again from its source.
    retry: () =>
      void runAction(
        (owner) => (canRestartFailedParts(job) ? restartFailedModelInstall(job.id, owner.signal) : resubmit(owner)),
        t('models.retryFailed')
      ),
  };
};
