import { Box, Flex, IconButton, Progress, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { isNil } from 'lodash-es';
import { useCallback, useMemo } from 'react';
import { PiXBold } from 'react-icons/pi';
import { useCancelModelInstallMutation } from 'services/api/endpoints/models';
import type { HFModelSource, LocalModelSource, ModelInstallJob, URLModelSource } from 'services/api/types';

import ModelInstallQueueBadge from './ModelInstallQueueBadge';

type ModelListItemProps = {
  installJob: ModelInstallJob;
};

const formatBytes = (bytes: number) => {
  const units = ['b', 'kb', 'mb', 'gb', 'tb'];

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

  const source = useMemo(() => {
    if (installJob.source.type === 'hf') {
      return installJob.source as HFModelSource;
    } else if (installJob.source.type === 'local') {
      return installJob.source as LocalModelSource;
    } else if (installJob.source.type === 'url') {
      return installJob.source as URLModelSource;
    } else {
      return installJob.source as LocalModelSource;
    }
  }, [installJob.source]);

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

  const modelName = useMemo(() => {
    switch (source.type) {
      case 'hf':
        return source.repo_id;
      case 'url':
        return source.url;
      case 'local':
        return source.path.split('\\').slice(-1)[0];
      default:
        return '';
    }
  }, [source]);

  const progressValue = useMemo(() => {
    if (isNil(installJob.bytes) || isNil(installJob.total_bytes)) {
      return null;
    }

    if (installJob.total_bytes === 0) {
      return 0;
    }

    return (installJob.bytes / installJob.total_bytes) * 100;
  }, [installJob.bytes, installJob.total_bytes]);

  const progressString = useMemo(() => {
    if (installJob.status !== 'downloading' || installJob.bytes === undefined || installJob.total_bytes === undefined) {
      return '';
    }
    return `${formatBytes(installJob.bytes)} / ${formatBytes(installJob.total_bytes)}`;
  }, [installJob.bytes, installJob.total_bytes, installJob.status]);

  return (
    <Flex gap="2" w="full" alignItems="center">
      <Tooltip label={modelName}>
        <Text width="30%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {modelName}
        </Text>
      </Tooltip>
      <Flex flexDir="column" flex={1}>
        <Tooltip label={progressString}>
          <Progress
            value={progressValue ?? 0}
            isIndeterminate={progressValue === null}
            aria-label={t('accessibility.invokeProgressBar')}
            h={2}
          />
        </Tooltip>
      </Flex>
      <Box minW="100px" textAlign="center">
        <ModelInstallQueueBadge status={installJob.status} errorReason={installJob.error_reason} />
      </Box>

      <Box minW="20px">
        {(installJob.status === 'downloading' ||
          installJob.status === 'waiting' ||
          installJob.status === 'running') && (
          <IconButton
            isRound={true}
            size="xs"
            tooltip={t('modelManager.cancel')}
            aria-label={t('modelManager.cancel')}
            icon={<PiXBold />}
            onClick={handleDeleteModelImport}
          />
        )}
      </Box>
    </Flex>
  );
};
