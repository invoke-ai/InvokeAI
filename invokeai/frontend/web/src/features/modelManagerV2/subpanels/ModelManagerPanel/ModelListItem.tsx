import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { ConfirmationAlertDialog, Flex, IconButton, Spacer, Text, useDisclosure } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectModelManagerV2Slice, setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelFormatBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelFormatBadge';
import { toast } from 'features/toast/toast';
import { filesize } from 'filesize';
import type { MouseEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteModelsMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import ModelImage from './ModelImage';

type ModelListItemProps = {
  model: AnyModelConfig;
};

const sx: SystemStyleObject = {
  paddingInline: 3,
  paddingBlock: 2,
  position: 'relative',
  rounded: 'base',
  '&:after,&:before': {
    content: `''`,
    position: 'absolute',
    pointerEvents: 'none',
  },
  '&:after': {
    h: '1px',
    bottom: '-0.5px',
    insetInline: 3,
    bg: 'base.850',
  },
  '&:before': {
    left: 1,
    w: 1,
    insetBlock: 2,
    rounded: 'base',
  },
  _hover: {
    bg: 'base.850',
    '& .delete-button': { opacity: 1 },
  },
  "&[aria-selected='false']:hover:before": { bg: 'base.750' },
  "&[aria-selected='true']": {
    bg: 'base.800',
    '& .delete-button': { opacity: 1 },
  },
  "&[aria-selected='true']:before": { bg: 'invokeBlue.300' },
};

const ModelListItem = ({ model }: ModelListItemProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectIsSelected = useMemo(
    () =>
      createSelector(
        selectModelManagerV2Slice,
        (modelManagerV2Slice) => modelManagerV2Slice.selectedModelKey === model.key
      ),
    [model.key]
  );
  const isSelected = useAppSelector(selectIsSelected);
  const [deleteModel] = useDeleteModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

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
  const handleModelDelete = useCallback(() => {
    deleteModel({ key: model.key })
      .unwrap()
      .then((_) => {
        toast({
          id: 'MODEL_DELETED',
          title: `${t('modelManager.modelDeleted')}: ${model.name}`,
          status: 'success',
        });
      })
      .catch((error) => {
        if (error) {
          toast({
            id: 'MODEL_DELETE_FAILED',
            title: `${t('modelManager.modelDeleteFailed')}: ${model.name}`,
            status: 'error',
          });
        }
      });
    dispatch(setSelectedModelKey(null));
  }, [deleteModel, model, dispatch, t]);

  return (
    <Flex
      sx={sx}
      aria-selected={isSelected}
      justifyContent="flex-start"
      w="full"
      alignItems="center"
      gap={2}
      cursor="pointer"
      onClick={handleSelectModel}
    >
      <Flex gap={2} w="full" h="full" minW={0}>
        <ModelImage image_url={model.cover_image} />
        <Flex alignItems="flex-start" flexDir="column" w="full" minW={0}>
          <Flex gap={2} w="full" alignItems="flex-start">
            <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
              {model.name}
            </Text>
            <Text variant="subtext" fontStyle="italic">
              {filesize(model.file_size)}
            </Text>
            <Spacer />
          </Flex>
          <Text variant="subtext" noOfLines={1}>
            {model.description || 'No Description'}
          </Text>
          <Flex gap={1} mt={1}>
            <ModelBaseBadge base={model.base} />
            <ModelFormatBadge format={model.format} />
          </Flex>
        </Flex>
      </Flex>
      <IconButton
        className="delete-button"
        onClick={onClickDeleteButton}
        icon={<PiTrashSimpleBold size={16} />}
        aria-label={t('modelManager.deleteConfig')}
        colorScheme="error"
        opacity={0}
        mt={1}
        alignSelf="flex-start"
      />
      <ConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('modelManager.deleteModel')}
        acceptCallback={handleModelDelete}
        acceptButtonText={t('modelManager.delete')}
        useInert={false}
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
