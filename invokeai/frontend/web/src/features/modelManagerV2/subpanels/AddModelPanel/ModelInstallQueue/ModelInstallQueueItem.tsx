import { Badge, Box, Flex, IconButton, Progress, Text, Tooltip } from '@invoke-ai/ui-library';
import { isNil } from 'es-toolkit/compat';
import { getApiErrorDetail } from 'features/modelManagerV2/util/getApiErrorDetail';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { memo, useCallback, useEffect, useMemo, useRef } from 'react';
import { PiArrowClockwiseBold, PiPauseBold, PiPlayBold, PiXBold } from 'react-icons/pi';
import {
  useCancelModelInstallMutation,
  usePauseModelInstallMutation,
  useRestartFailedModelInstallMutation,
  useRestartModelInstallFileMutation,
  useResumeModelInstallMutation,
} from 'services/api/endpoints/models';
import type { ModelInstallJob } from 'services/api/types';

import ModelInstallQueueBadge from './ModelInstallQueueBadge';

type ModelListItemProps = {
  installJob: ModelInstallJob;
};

const formatBytes = (bytes: number) => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];

  let i = 0;

  for (i; bytes >= 1024 && i < 4; i++) {
    bytes /= 1024;
  }

  return `${bytes.toFixed(2)} ${units[i]}`;
};

export const ModelInstallQueueItem = memo((props: ModelListItemProps) => {
  const { installJob } = props;

  const [deleteImportModel] = useCancelModelInstallMutation();
  const [pauseModelInstall] = usePauseModelInstallMutation();
  const [resumeModelInstall] = useResumeModelInstallMutation();
  const [restartFailedModelInstall] = useRestartFailedModelInstallMutation();
  const [restartModelInstallFile] = useRestartModelInstallFileMutation();
  const resumeFromScratchShown = useRef(false);

  const handleDeleteModelImport = useCallback(() => {
    deleteImportModel(installJob.id)
      .unwrap()
      .then((_) => {
        toast({
          id: 'MODEL_INSTALL_CANCELED',
          title: t('toast.modelImportCanceled'),
          status: 'success',
        });
      })
      .catch((error) => {
        toast({
          id: 'MODEL_INSTALL_CANCEL_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      });
  }, [deleteImportModel, installJob]);

  const handlePauseModelInstall = useCallback(() => {
    pauseModelInstall(installJob.id)
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_INSTALL_PAUSED',
          title: t('toast.modelDownloadPaused'),
          status: 'success',
        });
      })
      .catch((error) => {
        toast({
          id: 'MODEL_INSTALL_PAUSE_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      });
  }, [installJob, pauseModelInstall]);

  const hasRestartedFromScratch = useCallback((job: ModelInstallJob) => {
    return (
      job.download_parts?.some(
        (part) =>
          part.resume_from_scratch || (part.resume_message?.toLowerCase().includes('partial file missing') ?? false)
      ) ?? false
    );
  }, []);

  const handleResumeModelInstall = useCallback(() => {
    resumeModelInstall(installJob.id)
      .unwrap()
      .then((job) => {
        const restartedFromScratch = hasRestartedFromScratch(job);
        if (restartedFromScratch && !resumeFromScratchShown.current) {
          resumeFromScratchShown.current = true;
          toast({
            id: 'MODEL_INSTALL_RESTARTED_FROM_SCRATCH',
            title: t('toast.modelDownloadRestartedFromScratch'),
            status: 'warning',
          });
          return;
        }
        toast({
          id: 'MODEL_INSTALL_RESUMED',
          title: t('toast.modelDownloadResumed'),
          status: 'success',
        });
      })
      .catch((error) => {
        toast({
          id: 'MODEL_INSTALL_RESUME_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      });
  }, [hasRestartedFromScratch, installJob, resumeModelInstall]);

  const handleRestartFailed = useCallback(() => {
    restartFailedModelInstall(installJob.id)
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_INSTALL_RESTART_FAILED',
          title: t('toast.modelDownloadRestartFailed'),
          status: 'success',
        });
      })
      .catch((error) => {
        toast({
          id: 'MODEL_INSTALL_RESTART_FAILED_ERROR',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      });
  }, [installJob.id, restartFailedModelInstall]);

  const handleRestartFile = useCallback(
    (fileSource: string) => {
      restartModelInstallFile({ id: installJob.id, file_source: fileSource })
        .unwrap()
        .then(() => {
          toast({
            id: 'MODEL_INSTALL_RESTART_FILE',
            title: t('toast.modelDownloadRestartFile'),
            status: 'success',
          });
        })
        .catch((error) => {
          toast({
            id: 'MODEL_INSTALL_RESTART_FILE_ERROR',
            title: getApiErrorDetail(error),
            status: 'error',
          });
        });
    },
    [installJob.id, restartModelInstallFile]
  );

  const getRestartFileHandler = useCallback(
    (fileSource: string) => () => handleRestartFile(fileSource),
    [handleRestartFile]
  );

  const sourceLocation = useMemo(() => {
    switch (installJob.source.type) {
      case 'hf':
        return installJob.source.repo_id;
      case 'url':
        return installJob.source.url;
      case 'local':
        return installJob.source.path;
      default:
        return t('common.unknown');
    }
  }, [installJob.source]);

  const modelName = useMemo(() => {
    switch (installJob.source.type) {
      case 'hf': {
        const { repo_id, subfolder } = installJob.source;
        if (subfolder) {
          return `${repo_id}::${subfolder}`;
        }
        return repo_id;
      }
      case 'url':
        return installJob.source.url.split('/').slice(-1)[0] ?? t('common.unknown');
      case 'local':
        return installJob.source.path.split('\\').slice(-1)[0] ?? t('common.unknown');
      default:
        return t('common.unknown');
    }
  }, [installJob.source]);

  const progressValue = useMemo(() => {
    if (installJob.status === 'completed' || installJob.status === 'error' || installJob.status === 'cancelled') {
      return 100;
    }

    const parts = installJob.download_parts;
    if (parts && parts.length > 0) {
      const totalBytesFromParts = parts.reduce((sum, part) => sum + (part.total_bytes ?? 0), 0);
      const currentBytesFromParts = parts.reduce((sum, part) => sum + (part.bytes ?? 0), 0);
      const totalBytes = Math.max(totalBytesFromParts, installJob.total_bytes ?? 0);
      const currentBytes = Math.max(currentBytesFromParts, installJob.bytes ?? 0);
      if (totalBytes > 0) {
        return (currentBytes / totalBytes) * 100;
      }
      return 0;
    }

    if (!isNil(installJob.bytes) && !isNil(installJob.total_bytes) && installJob.total_bytes > 0) {
      return (installJob.bytes / installJob.total_bytes) * 100;
    }

    return null;
  }, [installJob.bytes, installJob.download_parts, installJob.status, installJob.total_bytes]);

  const restartRequiredParts = useMemo(() => {
    return installJob.download_parts?.filter((part) => part.resume_required || part.status === 'error') ?? [];
  }, [installJob.download_parts]);

  useEffect(() => {
    if (resumeFromScratchShown.current) {
      return;
    }
    const restartedFromScratch = hasRestartedFromScratch(installJob);
    if (restartedFromScratch) {
      resumeFromScratchShown.current = true;
      toast({
        id: 'MODEL_INSTALL_RESTARTED_FROM_SCRATCH_AUTO',
        title: t('toast.modelDownloadRestartedFromScratch'),
        status: 'warning',
      });
    }
  }, [hasRestartedFromScratch, installJob]);

  const hasRestartRequired = restartRequiredParts.length > 0;

  const showPause = installJob.status === 'downloading' || installJob.status === 'waiting';
  const showResume = installJob.status === 'paused' && !hasRestartRequired;
  const showCancel =
    installJob.status === 'downloading' ||
    installJob.status === 'waiting' ||
    installJob.status === 'running' ||
    installJob.status === 'paused';

  return (
    <>
      <Flex gap={1} w="full" alignItems="center">
        <Tooltip maxW={600} label={<TooltipLabel name={modelName} source={sourceLocation} installJob={installJob} />}>
          <Flex gap={3} w="full" alignItems="center">
            <Text w={96} whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
              {modelName}
            </Text>
            <Progress
              w="full"
              flexGrow={1}
              value={progressValue ?? 0}
              isIndeterminate={progressValue === null}
              aria-label={t('accessibility.invokeProgressBar')}
              h={2}
            />
            <ModelInstallQueueBadge status={installJob.status} />
          </Flex>
        </Tooltip>
        <Flex gap={1} alignItems="center" justifyContent="flex-end" minW="90px">
          {showResume && (
            <IconButton
              size="xs"
              tooltip={t('modelManager.resume')}
              aria-label={t('modelManager.resume')}
              icon={<PiPlayBold />}
              onClick={handleResumeModelInstall}
              variant="ghost"
            />
          )}
          {showPause && (
            <IconButton
              size="xs"
              tooltip={t('modelManager.pause')}
              aria-label={t('modelManager.pause')}
              icon={<PiPauseBold />}
              onClick={handlePauseModelInstall}
              variant="ghost"
            />
          )}
          {hasRestartRequired && (
            <IconButton
              size="xs"
              tooltip={t('modelManager.restartFailed')}
              aria-label={t('modelManager.restartFailed')}
              icon={<PiArrowClockwiseBold />}
              onClick={handleRestartFailed}
              variant="ghost"
            />
          )}
          {showCancel && (
            <IconButton
              size="xs"
              tooltip={t('modelManager.cancel')}
              aria-label={t('modelManager.cancel')}
              icon={<PiXBold />}
              onClick={handleDeleteModelImport}
              variant="ghost"
            />
          )}
          {!showResume && !showPause && !showCancel && <Box w="24px" />}
        </Flex>
      </Flex>
      {hasRestartRequired && (
        <Flex direction="column" gap={1} w="full" mt={1}>
          {restartRequiredParts.map((part) => {
            const fileName = part.source.split('/').slice(-1)[0] ?? t('common.unknown');
            const isResumeRequired = part.resume_required;
            return (
              <Flex key={part.source} gap={2} alignItems="center" wrap="wrap">
                <Text fontSize="xs" color="base.200" maxW="200px" noOfLines={1}>
                  {fileName}
                </Text>
                <Badge colorScheme={isResumeRequired ? 'orange' : 'red'} fontSize="10px">
                  {isResumeRequired ? t('modelManager.restartRequired') : t('queue.failed')}
                </Badge>
                <Text fontSize="xs" color="warning.400">
                  {isResumeRequired ? t('modelManager.resumeRefused') : t('queue.failed')}
                </Text>
                <IconButton
                  size="xs"
                  tooltip={t('modelManager.restartFile')}
                  aria-label={t('modelManager.restartFile')}
                  icon={<PiArrowClockwiseBold />}
                  onClick={getRestartFileHandler(part.source)}
                  variant="ghost"
                />
              </Flex>
            );
          })}
        </Flex>
      )}
    </>
  );
});

ModelInstallQueueItem.displayName = 'ModelInstallQueueItem';

type TooltipLabelProps = {
  installJob: ModelInstallJob;
  name: string;
  source: string;
};

const TooltipLabel = memo(({ name, source, installJob }: TooltipLabelProps) => {
  const progressString = useMemo(() => {
    if (installJob.status !== 'downloading') {
      return '';
    }
    const parts = installJob.download_parts;
    if (parts && parts.length > 0) {
      const totalBytesFromParts = parts.reduce((sum, part) => sum + (part.total_bytes ?? 0), 0);
      const currentBytesFromParts = parts.reduce((sum, part) => sum + (part.bytes ?? 0), 0);
      const totalBytes = Math.max(totalBytesFromParts, installJob.total_bytes ?? 0);
      const currentBytes = Math.max(currentBytesFromParts, installJob.bytes ?? 0);
      if (totalBytes > 0) {
        return `${formatBytes(currentBytes)} / ${formatBytes(totalBytes)}`;
      }
      return '';
    }
    if (!isNil(installJob.bytes) && !isNil(installJob.total_bytes) && installJob.total_bytes > 0) {
      return `${formatBytes(installJob.bytes)} / ${formatBytes(installJob.total_bytes)}`;
    }
    return '';
  }, [installJob.bytes, installJob.download_parts, installJob.total_bytes, installJob.status]);

  return (
    <>
      <Flex gap={3} justifyContent="space-between">
        <Text fontWeight="semibold">{name}</Text>
        {progressString && <Text>{progressString}</Text>}
      </Flex>
      <Text fontStyle="italic" wordBreak="break-all">
        {source}
      </Text>
      {installJob.error_reason && (
        <Text color="error.500">
          {t('queue.failed')}: {installJob.error}
        </Text>
      )}
    </>
  );
});

TooltipLabel.displayName = 'TooltipLabel';
