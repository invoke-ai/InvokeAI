import {
  Badge,
  Box,
  Button,
  ConfirmationAlertDialog,
  Flex,
  Icon,
  IconButton,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { IoWarning } from 'react-icons/io5';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteModelsMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import ModelImage from './ModelImage';

type ModelListItemProps = {
  model: AnyModelConfig;
};

const ModelListItem = (props: ModelListItemProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const [deleteModel] = useDeleteModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const { model } = props;

  const handleSelectModel = useCallback(() => {
    dispatch(setSelectedModelKey(model.key));
  }, [model.key, dispatch]);

  const isSelected = useMemo(() => {
    return selectedModelKey === model.key;
  }, [selectedModelKey, model.key]);

  const handleModelDelete = useCallback(() => {
    deleteModel({ key: model.key })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelDeleted')}: ${model.name}`,
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
                title: `${t('modelManager.modelDeleteFailed')}: ${model.name}`,
                status: 'error',
              })
            )
          );
        }
      });
    dispatch(setSelectedModelKey(null));
  }, [deleteModel, model, dispatch, t]);

  return (
    <Flex gap={2} alignItems="center" w="full">
      <ModelImage image_url={model.cover_image} />
      <Flex
        as={Button}
        isChecked={isSelected}
        variant={isSelected ? 'solid' : 'ghost'}
        justifyContent="start"
        p={2}
        borderRadius="base"
        w="full"
        alignItems="center"
        onClick={handleSelectModel}
      >
        <Flex gap={4} alignItems="center">
          <Badge minWidth={14} p={0.5} fontSize="sm" variant="solid">
            {MODEL_TYPE_SHORT_MAP[model.base as keyof typeof MODEL_TYPE_SHORT_MAP]}
          </Badge>
          <Tooltip label={model.description} placement="bottom">
            <Text>{model.name}</Text>
          </Tooltip>
          {model.format === 'checkpoint' && (
            <Tooltip label="Checkpoint">
              <Box>
                <Icon as={IoWarning} />
              </Box>
            </Tooltip>
          )}
        </Flex>
      </Flex>

      <IconButton
        onClick={onOpen}
        icon={<PiTrashSimpleBold />}
        aria-label={t('modelManager.deleteConfig')}
        colorScheme="error"
      />
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('modelManager.deleteModel')}
        acceptCallback={handleModelDelete}
        acceptButtonText={t('modelManager.delete')}
      >
        <Flex rowGap={4} flexDirection="column">
          <Text fontWeight="bold">{t('modelManager.deleteMsg1')}</Text>
          <Text>{t('modelManager.deleteMsg2')}</Text>
        </Flex>
      </ConfirmationAlertDialog>
    </Flex>
  );
};

export default memo(ModelListItem);
