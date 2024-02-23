import { Box, Flex, IconButton, Progress, Tag, Text, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiXBold } from 'react-icons/pi';
import { useDeleteModelImportMutation } from 'services/api/endpoints/models';
import type { ModelInstallJob, HFModelSource, LocalModelSource, URLModelSource } from 'services/api/types';
import ImportQueueBadge from './ImportQueueBadge';

type ModelListItemProps = {
  model: ModelInstallJob;
};

const formatBytes = (bytes: number) => {
  const units = ['b', 'kb', 'mb', 'gb', 'tb'];

  let i = 0;

  for (i; bytes >= 1024 && i < 4; i++) {
    bytes /= 1024;
  }

  return `${bytes.toFixed(2)} ${units[i]}`;
};

export const ImportQueueItem = (props: ModelListItemProps) => {
  const { model } = props;
  const dispatch = useAppDispatch();

  const [deleteImportModel] = useDeleteModelImportMutation();

  const source = useMemo(() => {
    if (model.source.type === 'hf') {
      return model.source as HFModelSource;
    } else if (model.source.type === 'local') {
      return model.source as LocalModelSource;
    } else if (model.source.type === 'url') {
      return model.source as URLModelSource;
    } else {
      return model.source as LocalModelSource;
    }
  }, [model.source]);

  const handleDeleteModelImport = useCallback(() => {
    deleteImportModel(model.id)
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
  }, [deleteImportModel, model, dispatch]);

  const modelName = useMemo(() => {
    switch (source.type) {
      case 'hf':
        return source.repo_id;
      case 'url':
        return source.url;
      case 'local':
        return source.path.substring(source.path.lastIndexOf('/') + 1);
      default:
        return '';
    }
  }, [source]);

  const progressValue = useMemo(() => {
    if (model.bytes === undefined || model.total_bytes === undefined) {
      return 0;
    }

    return (model.bytes / model.total_bytes) * 100;
  }, [model.bytes, model.total_bytes]);

  const progressString = useMemo(() => {
    if (model.status !== 'downloading' || model.bytes === undefined || model.total_bytes === undefined) {
      return '';
    }
    return `${formatBytes(model.bytes)} / ${formatBytes(model.total_bytes)}`;
  }, [model.bytes, model.total_bytes, model.status]);

  return (
    <Flex gap="2" w="full" alignItems="center" textAlign="center">
      <Tooltip label={modelName}>
        <Text w="30%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
          {modelName}
        </Text>
      </Tooltip>
      <Flex flexDir="column" w="50%">
        <Tooltip label={progressString}>
          <Progress
            value={progressValue}
            isIndeterminate={progressValue === undefined}
            aria-label={t('accessibility.invokeProgressBar')}
            h={2}
          />
        </Tooltip>
      </Flex>
      <Box w="15%">
        <ImportQueueBadge status={model.status} />
      </Box>

      <Box w="10%">
        {(model.status === 'downloading' || model.status === 'waiting') && (
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
