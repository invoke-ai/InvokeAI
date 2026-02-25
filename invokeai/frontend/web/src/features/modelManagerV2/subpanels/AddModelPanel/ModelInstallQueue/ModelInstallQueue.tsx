import { Box, Button, Flex, Heading } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { getApiErrorDetail } from 'features/modelManagerV2/util/getApiErrorDetail';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { memo, useCallback, useMemo, useRef, useState } from 'react';
import { PiPauseBold, PiPlayBold, PiXBold } from 'react-icons/pi';
import {
  useCancelModelInstallMutation,
  useListModelInstallsQuery,
  usePauseModelInstallMutation,
  usePruneCompletedModelInstallsMutation,
  useResumeModelInstallMutation,
} from 'services/api/endpoints/models';
import type { ModelInstallJob } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

import { ModelInstallQueueItem } from './ModelInstallQueueItem';

const hasRestartRequired = (job: ModelInstallJob) => {
  return job.download_parts?.some((part) => part.resume_required || part.status === 'error') ?? false;
};

export const ModelInstallQueue = memo(() => {
  const isConnected = useStore($isConnected);
  const { data } = useListModelInstallsQuery();
  const [bulkActionInProgress, setBulkActionInProgress] = useState<'pause' | 'resume' | 'cancel' | null>(null);
  const bulkActionLockRef = useRef(false);

  const [cancelModelInstall] = useCancelModelInstallMutation();
  const [pauseModelInstall] = usePauseModelInstallMutation();
  const [resumeModelInstall] = useResumeModelInstallMutation();
  const [_pruneCompletedModelInstalls, { isLoading: isPruning }] = usePruneCompletedModelInstallsMutation();

  const installActionIds = useMemo(() => {
    const pauseable: number[] = [];
    const resumable: number[] = [];
    const cancelable: number[] = [];

    for (const model of data ?? []) {
      if (model.status === 'downloading' || model.status === 'waiting') {
        pauseable.push(model.id);
      }

      if (model.status === 'paused') {
        cancelable.push(model.id);
        if (!hasRestartRequired(model)) {
          resumable.push(model.id);
        }
        continue;
      }

      if (model.status === 'running') {
        cancelable.push(model.id);
      }
    }

    return {
      pauseable,
      resumable,
      cancelable: cancelable.concat(pauseable),
    } as const;
  }, [data]);

  const {
    pauseable: pauseableInstallIds,
    resumable: resumableInstallIds,
    cancelable: cancelableInstallIds,
  } = installActionIds;

  const runBulkAction = useCallback(
    async (action: 'pause' | 'resume' | 'cancel', installIds: number[]) => {
      if (installIds.length === 0 || bulkActionLockRef.current || isPruning) {
        return;
      }

      bulkActionLockRef.current = true;
      setBulkActionInProgress(action);

      try {
        const results = await Promise.allSettled(
          installIds.map((id) => {
            if (action === 'pause') {
              return pauseModelInstall(id).unwrap();
            }
            if (action === 'resume') {
              return resumeModelInstall(id).unwrap();
            }
            return cancelModelInstall(id).unwrap();
          })
        );

        const hasSucceeded = results.some((result) => result.status === 'fulfilled');
        const firstError = results.find((result) => result.status === 'rejected') as PromiseRejectedResult | undefined;

        if (hasSucceeded) {
          const title =
            action === 'pause'
              ? t('toast.modelDownloadPaused')
              : action === 'resume'
                ? t('toast.modelDownloadResumed')
                : t('toast.modelImportCanceled');

          toast({
            id: `MODEL_INSTALL_QUEUE_${action.toUpperCase()}_ALL`,
            title,
            status: 'success',
          });
        }

        if (firstError) {
          toast({
            id: `MODEL_INSTALL_QUEUE_${action.toUpperCase()}_ALL_FAILED`,
            title: getApiErrorDetail(firstError.reason),
            status: 'error',
          });
        }
      } finally {
        bulkActionLockRef.current = false;
        setBulkActionInProgress(null);
      }
    },
    [cancelModelInstall, isPruning, pauseModelInstall, resumeModelInstall]
  );

  const pruneCompletedModelInstalls = useCallback(async () => {
    if (bulkActionLockRef.current) {
      return;
    }

    try {
      await _pruneCompletedModelInstalls().unwrap();
      toast({
        id: 'MODEL_INSTALL_QUEUE_PRUNED',
        title: t('toast.prunedQueue'),
        status: 'success',
      });
    } catch (error) {
      toast({
        id: 'MODEL_INSTALL_QUEUE_PRUNE_FAILED',
        title: getApiErrorDetail(error),
        status: 'error',
      });
    }
  }, [_pruneCompletedModelInstalls]);

  const hasPauseableInstalls = pauseableInstallIds.length > 0;
  const hasResumableInstalls = resumableInstallIds.length > 0;
  const hasCancelableInstalls = cancelableInstallIds.length > 0;
  const showResumeAll = !hasPauseableInstalls && hasResumableInstalls;
  const pauseResumeAvailable = hasPauseableInstalls || hasResumableInstalls;

  const pruneAvailable = useMemo(() => {
    return data?.some(
      (model) => model.status === 'cancelled' || model.status === 'error' || model.status === 'completed'
    );
  }, [data]);

  const pauseResumeLabel = showResumeAll ? t('modelManager.resumeAll') : t('modelManager.pauseAll');
  const pauseResumeTooltip = showResumeAll ? t('modelManager.resumeAllTooltip') : t('modelManager.pauseAllTooltip');

  const pauseOrResumeAll = useCallback(() => {
    if (showResumeAll) {
      void runBulkAction('resume', resumableInstallIds);
      return;
    }

    void runBulkAction('pause', pauseableInstallIds);
  }, [pauseableInstallIds, resumableInstallIds, runBulkAction, showResumeAll]);

  const cancelAll = useCallback(() => {
    void runBulkAction('cancel', cancelableInstallIds);
  }, [cancelableInstallIds, runBulkAction]);

  const isBulkActionRunning = bulkActionInProgress !== null;

  return (
    <Flex flexDir="column" p={3} h="full" gap={3}>
      <Flex justifyContent="space-between" alignItems="center">
        <Flex alignItems="center" gap={2}>
          <Heading size="sm">{t('modelManager.installQueue')}</Heading>
          {!isConnected && (
            <Box layerStyle="first" px={2} py={0.5} borderRadius="base">
              <Heading size="sm" color="error.300">
                {t('modelManager.backendDisconnected')}
              </Heading>
            </Box>
          )}
        </Flex>
        <Flex gap={2} alignItems="center">
          <Button
            size="sm"
            leftIcon={showResumeAll ? <PiPlayBold /> : <PiPauseBold />}
            isDisabled={!pauseResumeAvailable || isBulkActionRunning || isPruning}
            isLoading={bulkActionInProgress === 'pause' || bulkActionInProgress === 'resume'}
            onClick={pauseOrResumeAll}
            tooltip={pauseResumeTooltip}
          >
            {pauseResumeLabel}
          </Button>
          <Button
            size="sm"
            leftIcon={<PiXBold />}
            isDisabled={!hasCancelableInstalls || isBulkActionRunning || isPruning}
            isLoading={bulkActionInProgress === 'cancel'}
            onClick={cancelAll}
            tooltip={t('modelManager.cancelAllTooltip')}
          >
            {t('modelManager.cancelAll')}
          </Button>
          <Button
            size="sm"
            isDisabled={!pruneAvailable || isBulkActionRunning}
            isLoading={isPruning}
            onClick={pruneCompletedModelInstalls}
            tooltip={t('modelManager.pruneTooltip')}
          >
            {t('modelManager.prune')}
          </Button>
        </Flex>
      </Flex>
      <Box layerStyle="first" p={3} borderRadius="base" w="full" h="full">
        <ScrollableContent>
          <Flex flexDir="column-reverse" gap="2" w="full">
            {data?.map((model) => (
              <ModelInstallQueueItem key={model.id} installJob={model} />
            ))}
          </Flex>
        </ScrollableContent>
      </Box>
    </Flex>
  );
});

ModelInstallQueue.displayName = 'ModelInstallQueue';
