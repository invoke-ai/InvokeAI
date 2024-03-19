import { Flex, IconButton, Progress, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { PiXBold } from 'react-icons/pi';
import { useCancelModelInstallMutation } from 'services/api/endpoints/models';
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

export const ModelInstallQueueItem = (props: ModelListItemProps) => {
  const { installJob } = props;
  const dispatch = useAppDispatch();

  const [deleteImportModel] = useCancelModelInstallMutation();

  const handleDeleteModelImport = useCallback(() => {
    deleteImportModel(installJob.id)
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.modelImportCanceled'),
              status: 'success',
            })
          )
        );
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: `${error.data.detail} `,
                status: 'error',
              })
            )
          );
        }
      });
  }, [deleteImportModel, installJob, dispatch]);

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
      case 'hf':
        return installJob.source.repo_id;
      case 'url':
        return installJob.source.url.split('/').slice(-1)[0] ?? t('common.unknown');
      case 'local':
        return installJob.source.path.split('\\').slice(-1)[0] ?? t('common.unknown');
      default:
        return t('common.unknown');
    }
  }, [installJob.source]);

  const progressValue = useMemo(() => {
    if (isNil(installJob.bytes) || isNil(installJob.total_bytes)) {
      return null;
    }

    if (installJob.total_bytes === 0) {
      return 0;
    }

    return (installJob.bytes / installJob.total_bytes) * 100;
  }, [installJob.bytes, installJob.total_bytes]);

  return (
    <Flex gap={3} w="full" alignItems="center">
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
      <IconButton
        isDisabled={
          installJob.status !== 'downloading' && installJob.status !== 'waiting' && installJob.status !== 'running'
        }
        size="xs"
        tooltip={t('modelManager.cancel')}
        aria-label={t('modelManager.cancel')}
        icon={<PiXBold />}
        onClick={handleDeleteModelImport}
        variant="ghost"
      />
    </Flex>
  );
};

type TooltipLabelProps = {
  installJob: ModelInstallJob;
  name: string;
  source: string;
};

const TooltipLabel = ({ name, source, installJob }: TooltipLabelProps) => {
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
};
