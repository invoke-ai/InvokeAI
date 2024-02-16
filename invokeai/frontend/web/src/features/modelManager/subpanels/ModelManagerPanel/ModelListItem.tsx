import {
  Badge,
  Button,
  ConfirmationAlertDialog,
  Flex,
  IconButton,
  Text,
  Tooltip,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import type { LoRAConfig, MainModelConfig } from 'services/api/endpoints/models';
import { useDeleteLoRAModelsMutation, useDeleteMainModelsMutation } from 'services/api/endpoints/models';

type ModelListItemProps = {
  model: MainModelConfig | LoRAConfig;
  isSelected: boolean;
  setSelectedModelId: (v: string | undefined) => void;
};

const ModelListItem = (props: ModelListItemProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const [deleteMainModel] = useDeleteMainModelsMutation();
  const [deleteLoRAModel] = useDeleteLoRAModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const { model, isSelected, setSelectedModelId } = props;

  const handleSelectModel = useCallback(() => {
    setSelectedModelId(model.id);
  }, [model.id, setSelectedModelId]);

  const handleModelDelete = useCallback(() => {
    const method = {
      main: deleteMainModel,
      lora: deleteLoRAModel,
    }[model.model_type];

    method(model)
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelDeleted')}: ${model.model_name}`,
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
                title: `${t('modelManager.modelDeleteFailed')}: ${model.model_name}`,
                status: 'error',
              })
            )
          );
        }
      });
    setSelectedModelId(undefined);
  }, [deleteMainModel, deleteLoRAModel, model, setSelectedModelId, dispatch, t]);

  return (
    <Flex gap={2} alignItems="center" w="full">
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
            {MODEL_TYPE_SHORT_MAP[model.base_model as keyof typeof MODEL_TYPE_SHORT_MAP]}
          </Badge>
          <Tooltip label={model.description} placement="bottom">
            <Text>{model.model_name}</Text>
          </Tooltip>
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
