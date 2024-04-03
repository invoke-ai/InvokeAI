import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { ConfirmationAlertDialog, Flex, IconButton, Spacer, Text, useDisclosure } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelFormatBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelFormatBadge';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteModelsMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import ModelImage, { MODEL_IMAGE_THUMBNAIL_SIZE } from './ModelImage';

type ModelListItemProps = {
  model: AnyModelConfig;
};

const sx: SystemStyleObject = {
  _hover: { bg: 'base.700' },
  "&[aria-selected='true']": { bg: 'base.700' },
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

  const onClickDeleteButton = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      onOpen();
    },
    [onOpen]
  );

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
    <Flex
      sx={sx}
      aria-selected={isSelected}
      justifyContent="flex-start"
      p={2}
      borderRadius="base"
      w="full"
      alignItems="center"
      gap={2}
      cursor="pointer"
      onClick={handleSelectModel}
    >
      <Flex gap={2} w="full" h="full" minW={0}>
        <ModelImage image_url={model.cover_image} />
        <Flex gap={1} alignItems="flex-start" flexDir="column" w="full" minW={0}>
          <Flex gap={2} w="full" alignItems="flex-start">
            <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
              {model.name}
            </Text>
            <Spacer />
          </Flex>
          <Text variant="subtext" noOfLines={1}>
            {model.description || 'No Description'}
          </Text>
        </Flex>
        <Flex
          h={MODEL_IMAGE_THUMBNAIL_SIZE}
          flexDir="column"
          alignItems="flex-end"
          justifyContent="space-between"
          gap={2}
        >
          <ModelBaseBadge base={model.base} />
          <ModelFormatBadge format={model.format} />
        </Flex>
      </Flex>
      <IconButton
        onClick={onClickDeleteButton}
        icon={<PiTrashSimpleBold size={16} />}
        aria-label={t('modelManager.deleteConfig')}
        colorScheme="error"
        h={MODEL_IMAGE_THUMBNAIL_SIZE}
        w={MODEL_IMAGE_THUMBNAIL_SIZE}
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
