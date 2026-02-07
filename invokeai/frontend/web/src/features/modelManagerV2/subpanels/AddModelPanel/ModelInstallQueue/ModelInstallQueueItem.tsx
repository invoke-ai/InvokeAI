import { Box, Flex, IconButton, Progress, Text, Tooltip } from '@invoke-ai/ui-library';
import { isNil } from 'es-toolkit/compat';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { memo, useCallback, useMemo } from 'react';
import { PiPauseBold, PiPlayBold, PiXBold } from 'react-icons/pi';
import {
  useCancelModelInstallMutation,
  usePauseModelInstallMutation,
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
        if (error) {
          toast({
            id: 'MODEL_INSTALL_CANCEL_FAILED',
            title: `${error.data.detail} `,
            status: 'error',
          });
        }
      });
  }, [deleteImportModel, installJob]);

  const handlePauseModelInstall = useCallback(() => {
    pauseModelInstall(installJob.id)
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_INSTALL_PAUSED',
          title: t('modelManager.pause'),
          status: 'success',
        });
      })
      .catch((error) => {
        if (error) {
          toast({
            id: 'MODEL_INSTALL_PAUSE_FAILED',
            title: `${error.data.detail} `,
            status: 'error',
          });
        }
      });
  }, [installJob, pauseModelInstall]);

  const handleResumeModelInstall = useCallback(() => {
    resumeModelInstall(installJob.id)
      .unwrap()
      .then(() => {
        toast({
          id: 'MODEL_INSTALL_RESUMED',
          title: t('modelManager.resume'),
          status: 'success',
        });
      })
      .catch((error) => {
        if (error) {
          toast({
            id: 'MODEL_INSTALL_RESUME_FAILED',
            title: `${error.data.detail} `,
            status: 'error',
          });
        }
      });
  }, [installJob, resumeModelInstall]);

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

    if (isNil(installJob.bytes) || isNil(installJob.total_bytes)) {
      return null;
    }

    if (installJob.total_bytes === 0) {
      return 0;
    }

    return (installJob.bytes / installJob.total_bytes) * 100;
  }, [installJob.bytes, installJob.status, installJob.total_bytes]);

  const showPause = installJob.status === 'downloading' || installJob.status === 'waiting';
  const showResume = installJob.status === 'paused';
  const showCancel =
    installJob.status === 'downloading' ||
    installJob.status === 'waiting' ||
    installJob.status === 'running' ||
    installJob.status === 'paused';

  return (
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
      <Flex gap={1} alignItems="center" justifyContent="flex-end" minW="64px">
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
    if (installJob.status !== 'downloading' || installJob.bytes === undefined || installJob.total_bytes === undefined) {
      return '';
    }
    return `${formatBytes(installJob.bytes)} / ${formatBytes(installJob.total_bytes)}`;
  }, [installJob.bytes, installJob.total_bytes, installJob.status]);

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
