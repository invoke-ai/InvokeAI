import { Box, Flex, IconButton, Progress, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import { useCallback, useMemo } from 'react';
import { PiXBold } from 'react-icons/pi';
import { useDeleteModelImportMutation } from 'services/api/endpoints/models';
import type { ImportModelConfig } from 'services/api/types';

type ModelListItemProps = {
  model: ImportModelConfig;
};

export const ImportQueueModel = (props: ModelListItemProps) => {
  const { model } = props;
  const dispatch = useAppDispatch();

  const [deleteImportModel] = useDeleteModelImportMutation();

  const handleDeleteModelImport = useCallback(() => {
    deleteImportModel({ key: model.id })
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

  const formatBytes = (bytes: number) => {
    const units = ['b', 'kb', 'mb', 'gb', 'tb'];

    let i = 0;

    for (i; bytes >= 1024 && i < 4; i++) {
      bytes /= 1024;
    }

    return `${bytes.toFixed(2)} ${units[i]}`;
  };

  const modelName = useMemo(() => {
    return model.source.repo_id || model.source.url || model.source.path.substring(model.source.path.lastIndexOf('/') + 1);
  }, [model.source]);

  const progressValue = useMemo(() => {
    return (model.bytes / model.total_bytes) * 100;
  }, [model.bytes, model.total_bytes]);

  const progressString = useMemo(() => {
    if (model.status !== 'downloading') {
      return '--';
    }
    return `${formatBytes(model.bytes)} / ${formatBytes(model.total_bytes)}`;
  }, [model.bytes, model.total_bytes, model.status]);

  return (
    <Flex gap="2" w="full" alignItems="center" textAlign="center">
      <Text w="20%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
        {modelName}
      </Text>
      <Progress
        value={progressValue}
        isIndeterminate={progressValue === undefined}
        aria-label={t('accessibility.invokeProgressBar')}
        h={2}
        w="50%"
      />
      <Text minW="20%" fontSize="xs" w="20%">
        {progressString}
      </Text>
      <Text w="15%">{model.status[0].toUpperCase() + model.status.slice(1)}</Text>
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
