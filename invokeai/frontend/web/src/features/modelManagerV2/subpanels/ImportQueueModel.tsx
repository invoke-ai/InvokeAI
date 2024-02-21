import {
  Flex,
  IconButton,
  Progress,
  Text,
} from '@invoke-ai/ui-library';
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

  const progressValue = useMemo(() => {
    return (model.bytes / model.total_bytes) * 100;
  }, [model.bytes, model.total_bytes]);

  return (
    <Flex gap="3" w="full" alignItems="center" textAlign="center">
      <Text w="20%" whiteSpace="nowrap" overflow="hidden" textOverflow="ellipsis">
        {model.source.repo_id}
      </Text>
      <Progress
        value={progressValue}
        isIndeterminate={progressValue === undefined}
        aria-label={t('accessibility.invokeProgressBar')}
        h={2}
        w="50%"
      />
      <Text w="20%">{model.status}</Text>
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
    </Flex>
  );
};
