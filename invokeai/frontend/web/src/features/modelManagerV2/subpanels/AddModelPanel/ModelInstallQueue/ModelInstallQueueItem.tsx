import type { SystemStyleObject } from '@invoke-ai/ui-library';
import {
  Badge,
  Box,
  Button,
  CircularProgress,
  Flex,
  Icon,
  IconButton,
  Td,
  Text,
  Tooltip,
  Tr,
} from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { isNil } from 'es-toolkit/compat';
import { getApiErrorDetail } from 'features/modelManagerV2/util/getApiErrorDetail';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  PiArrowClockwiseBold,
  PiCheckBold,
  PiMinusBold,
  PiPauseFill,
  PiPlayFill,
  PiWarningBold,
  PiWarningDiamondBold,
  PiWarningFill,
  PiXBold,
} from 'react-icons/pi';
import {
  useCancelModelInstallMutation,
  usePauseModelInstallMutation,
  useRestartFailedModelInstallMutation,
  useRestartModelInstallFileMutation,
  useResumeModelInstallMutation,
} from 'services/api/endpoints/models';
import type { ModelInstallJob, ModelInstallStatus } from 'services/api/types';
import { $isConnected } from 'services/events/stores';

import { ModelInstallQueueBadge } from './ModelInstallQueueBadge';

type ModelListItemProps = {
  installJob: ModelInstallJob;
};

type QueueItemAction = 'cancel' | 'pause' | 'resume' | 'restartFailed' | 'restartFile';
type OptimisticStatusState = {
  status: ModelInstallStatus;
  previousStatus: ModelInstallStatus | undefined;
};

const OPTIMISTIC_STATUS_BY_ACTION: Record<QueueItemAction, ModelInstallStatus> = {
  cancel: 'cancelled',
  pause: 'paused',
  resume: 'waiting',
  restartFailed: 'waiting',
  restartFile: 'waiting',
};

const isRestartableStatus = (status?: ModelInstallStatus) => status === 'paused' || status === 'error';

const formatBytes = (bytes: number) => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];

  let i = 0;

  for (i; bytes >= 1024 && i < 4; i++) {
    bytes /= 1024;
  }

  return `${bytes.toFixed(2)} ${units[i]}`;
};

const ProgressColumnSx: SystemStyleObject = {
  alignItems: 'center',
  justifyContent: 'center',
};

const ModelInfoColumnSx: SystemStyleObject = {
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 0.5,
};

const BadgesColumnSx: SystemStyleObject = {
  gap: 1,
  alignItems: 'flex-start',
  flexWrap: 'wrap',
};

const ActionsColumnSx: SystemStyleObject = {
  gap: 2,
  alignItems: 'flex-start',
  justifyContent: 'flex-end',
};

const CircularProgressSx: SystemStyleObject = {
  '.chakra-progress__track': {
    stroke: 'base.600',
  },
  '.chakra-progress__indicator': {
    stroke: 'blue.300',
  },
};

export const ModelInstallQueueItem = memo((props: ModelListItemProps) => {
  const { t } = useTranslation();
  const { installJob } = props;

  const [deleteImportModel] = useCancelModelInstallMutation();
  const [pauseModelInstall] = usePauseModelInstallMutation();
  const [resumeModelInstall] = useResumeModelInstallMutation();
  const [restartFailedModelInstall] = useRestartFailedModelInstallMutation();
  const [restartModelInstallFile] = useRestartModelInstallFileMutation();
  const [actionInFlight, setActionInFlight] = useState<QueueItemAction | null>(null);
  const [optimisticStatus, setOptimisticStatus] = useState<OptimisticStatusState | null>(null);
  const actionInFlightRef = useRef<QueueItemAction | null>(null);
  const resumeFromScratchShown = useRef(false);
  const isConnected = useStore($isConnected);

  useEffect(() => {
    if (!optimisticStatus) {
      return;
    }

    if (installJob.status !== optimisticStatus.previousStatus) {
      setOptimisticStatus(null);
    }
  }, [installJob.status, optimisticStatus]);

  const withRowActionLock = useCallback(
    async (action: QueueItemAction, previousStatus: ModelInstallStatus | undefined, fn: () => Promise<void>) => {
      if (actionInFlightRef.current) {
        return;
      }

      actionInFlightRef.current = action;
      setActionInFlight(action);
      setOptimisticStatus({ status: OPTIMISTIC_STATUS_BY_ACTION[action], previousStatus });

      try {
        await fn();
      } finally {
        actionInFlightRef.current = null;
        setActionInFlight(null);
      }
    },
    []
  );

  const handleDeleteModelImport = useCallback(() => {
    void withRowActionLock('cancel', installJob.status, async () => {
      try {
        await deleteImportModel(installJob.id).unwrap();
        toast({
          id: 'MODEL_INSTALL_CANCELED',
          title: t('toast.modelImportCanceled'),
          status: 'success',
        });
      } catch (error) {
        setOptimisticStatus(null);
        toast({
          id: 'MODEL_INSTALL_CANCEL_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      }
    });
  }, [deleteImportModel, installJob.id, installJob.status, t, withRowActionLock]);

  const handlePauseModelInstall = useCallback(() => {
    void withRowActionLock('pause', installJob.status, async () => {
      try {
        await pauseModelInstall(installJob.id).unwrap();
        toast({
          id: 'MODEL_INSTALL_PAUSED',
          title: t('toast.modelDownloadPaused'),
          status: 'success',
        });
      } catch (error) {
        setOptimisticStatus(null);
        toast({
          id: 'MODEL_INSTALL_PAUSE_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      }
    });
  }, [installJob.id, installJob.status, pauseModelInstall, t, withRowActionLock]);

  const hasRestartedFromScratch = useCallback((job: ModelInstallJob) => {
    return (
      job.download_parts?.some(
        (part) =>
          part.resume_from_scratch || (part.resume_message?.toLowerCase().includes('partial file missing') ?? false)
      ) ?? false
    );
  }, []);

  const handleResumeModelInstall = useCallback(() => {
    void withRowActionLock('resume', installJob.status, async () => {
      try {
        const job = await resumeModelInstall(installJob.id).unwrap();
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
      } catch (error) {
        setOptimisticStatus(null);
        toast({
          id: 'MODEL_INSTALL_RESUME_FAILED',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      }
    });
  }, [hasRestartedFromScratch, installJob.id, installJob.status, resumeModelInstall, t, withRowActionLock]);

  const handleRestartFailed = useCallback(() => {
    void withRowActionLock('restartFailed', installJob.status, async () => {
      try {
        await restartFailedModelInstall(installJob.id).unwrap();
        toast({
          id: 'MODEL_INSTALL_RESTART_FAILED',
          title: t('toast.modelDownloadRestartFailed'),
          status: 'success',
        });
      } catch (error) {
        setOptimisticStatus(null);
        toast({
          id: 'MODEL_INSTALL_RESTART_FAILED_ERROR',
          title: getApiErrorDetail(error),
          status: 'error',
        });
      }
    });
  }, [installJob.id, installJob.status, restartFailedModelInstall, t, withRowActionLock]);

  const handleRestartFile = useCallback(
    (fileSource: string) => {
      void withRowActionLock('restartFile', installJob.status, async () => {
        try {
          await restartModelInstallFile({ id: installJob.id, file_source: fileSource }).unwrap();
          toast({
            id: 'MODEL_INSTALL_RESTART_FILE',
            title: t('toast.modelDownloadRestartFile'),
            status: 'success',
          });
        } catch (error) {
          setOptimisticStatus(null);
          toast({
            id: 'MODEL_INSTALL_RESTART_FILE_ERROR',
            title: getApiErrorDetail(error),
            status: 'error',
          });
        }
      });
    },
    [installJob.id, installJob.status, restartModelInstallFile, t, withRowActionLock]
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
  }, [installJob.source, t]);

  const displayStatus = optimisticStatus?.status ?? installJob.status;

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
        return installJob.source.path.split(/[/\\]/).slice(-1)[0] ?? t('common.unknown');
      default:
        return t('common.unknown');
    }
  }, [installJob.source, t]);

  const progressValue = useMemo(() => {
    if (displayStatus === 'completed' || displayStatus === 'error' || displayStatus === 'cancelled') {
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
  }, [displayStatus, installJob.bytes, installJob.download_parts, installJob.total_bytes]);

  const progressTooltip = useMemo(() => {
    if (displayStatus !== 'downloading' && displayStatus !== 'downloads_done') {
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
  }, [displayStatus, installJob.bytes, installJob.download_parts, installJob.total_bytes]);

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
  }, [hasRestartedFromScratch, installJob, t]);

  const hasRestartRequired = isRestartableStatus(displayStatus) && restartRequiredParts.length > 0;

  const canPause = displayStatus === 'downloading' || displayStatus === 'waiting';
  const canResume = displayStatus === 'paused' && !hasRestartRequired;
  const canCancel =
    displayStatus === 'downloading' ||
    displayStatus === 'waiting' ||
    displayStatus === 'downloads_done' ||
    displayStatus === 'running' ||
    displayStatus === 'paused';

  const isActiveInstall =
    displayStatus === 'downloading' ||
    displayStatus === 'waiting' ||
    displayStatus === 'downloads_done' ||
    displayStatus === 'running';

  const hasVisibleError = displayStatus === 'error' ? installJob.error : null;
  const isActionInFlight = actionInFlight !== null;

  const showDisconnectedIndicator = !isConnected && isActiveInstall;

  return (
    <Tr>
      {/* Progress */}
      <Td>
        <Flex sx={ProgressColumnSx}>
          {isActiveInstall ? (
            <Tooltip label={progressTooltip} isDisabled={!progressTooltip} hasArrow openDelay={0}>
              <CircularProgress
                size="20px"
                value={progressValue ?? 0}
                isIndeterminate={
                  !isConnected ||
                  progressValue === null ||
                  displayStatus === 'waiting' ||
                  displayStatus === 'downloads_done' ||
                  displayStatus === 'running'
                }
                aria-label={t('accessibility.invokeProgressBar')}
                sx={CircularProgressSx}
                thickness={12}
              />
            </Tooltip>
          ) : displayStatus === 'paused' ? (
            <Flex sx={{ color: 'orange.300' }}>
              <PiPauseFill size={16} />
            </Flex>
          ) : displayStatus === 'cancelled' ? (
            <Flex sx={{ color: 'orange.200' }}>
              <PiMinusBold size={16} />
            </Flex>
          ) : displayStatus === 'error' ? (
            <Flex sx={{ color: 'red.300' }}>
              <PiXBold size={16} />
            </Flex>
          ) : (
            <Flex sx={{ color: 'green.300' }}>
              <PiCheckBold size={16} />
            </Flex>
          )}
        </Flex>
      </Td>

      {/* Model Info */}
      <Td>
        <Flex sx={ModelInfoColumnSx}>
          <Text fontWeight="semibold">{modelName}</Text>
          <Text fontStyle="italic" fontSize="2xs">
            {sourceLocation}
          </Text>
          {hasRestartRequired && (
            <Flex direction="column" gap={1} w="full" mt={1}>
              {restartRequiredParts.map((part) => {
                const fileName = part.source.split(/[/\\]/).slice(-1)[0] ?? t('common.unknown');
                const isResumeRequired = part.resume_required;
                return (
                  <Flex key={part.source} gap={2} alignItems="center" wrap="wrap" p={2} bg="base.800" borderRadius="md">
                    <Icon
                      as={isResumeRequired ? PiWarningFill : PiWarningDiamondBold}
                      color={isResumeRequired ? 'orange.500' : 'red.500'}
                    />
                    <Text fontSize="xs" color="base.200" maxW="200px" noOfLines={1} title={fileName}>
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
                      ml="auto"
                      isDisabled={isActionInFlight}
                    />
                  </Flex>
                );
              })}
            </Flex>
          )}
        </Flex>
      </Td>

      {/* Status */}
      <Td>
        <Flex sx={BadgesColumnSx}>
          {showDisconnectedIndicator && (
            <Tooltip label={t('modelManager.backendDisconnected')}>
              <Box padding={1}>
                <Icon as={PiWarningBold} color="error.300" />
              </Box>
            </Tooltip>
          )}
          <ModelInstallQueueBadge status={displayStatus} label={hasVisibleError} />
          {hasRestartRequired && (
            <Tooltip label={t('modelManager.restartRequiredTooltip')}>
              <Box>
                <Badge colorScheme="red">{t('modelManager.restartRequired')}</Badge>
              </Box>
            </Tooltip>
          )}
        </Flex>
      </Td>

      {/* Actions */}
      <Td textAlign="right" minWidth={130}>
        <Flex sx={ActionsColumnSx}>
          {/* Pause/Resume installatino */}
          {(canResume || canPause) && (
            <Button
              size="sm"
              tooltip={canResume ? t('modelManager.resume') : t('modelManager.pause')}
              leftIcon={canResume ? <PiPlayFill /> : <PiPauseFill />}
              onClick={canResume ? handleResumeModelInstall : handlePauseModelInstall}
              variant={canResume ? 'solid' : 'outline'}
              isDisabled={isActionInFlight}
            >
              {canResume ? t('modelManager.resume') : t('modelManager.pause')}
            </Button>
          )}

          {/* Restart installation if required */}
          {hasRestartRequired && (
            <Button
              tooltip={t('modelManager.restartFailed')}
              size="sm"
              leftIcon={<PiArrowClockwiseBold />}
              onClick={handleRestartFailed}
              colorScheme="error"
              variant="ghost"
              isDisabled={isActionInFlight}
            >
              {t('modelManager.restartFailed')}
            </Button>
          )}

          {/* Cancel installation */}
          {canCancel && (
            <IconButton
              tooltip={t('modelManager.cancel')}
              icon={<PiXBold />}
              aria-label={t('modelManager.cancel')}
              onClick={handleDeleteModelImport}
              size="sm"
              colorScheme="error"
              isDisabled={isActionInFlight}
            />
          )}

          {!canCancel && !canPause && !canResume && (
            // TODO: Add an individual prune action here?
            <Text fontSize="2xs">No actions available</Text>
          )}
        </Flex>
      </Td>
    </Tr>
  );
});

ModelInstallQueueItem.displayName = 'ModelInstallQueueItem';
