import { DeleteIcon } from '@chakra-ui/icons';
import { Badge, Flex, useDisclosure } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { MODEL_TYPE_SHORT_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type {
  LoRAModelConfigEntity,
  MainModelConfigEntity,
  OnnxModelConfigEntity,
} from 'services/api/endpoints/models';
import {
  useDeleteLoRAModelsMutation,
  useDeleteMainModelsMutation,
} from 'services/api/endpoints/models';

type ModelListItemProps = {
  model: MainModelConfigEntity | OnnxModelConfigEntity | LoRAModelConfigEntity;
  isSelected: boolean;
  setSelectedModelId: (v: string | undefined) => void;
};

export default function ModelListItem(props: ModelListItemProps) {
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
      onnx: deleteMainModel,
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
                title: `${t('modelManager.modelDeleteFailed')}: ${
                  model.model_name
                }`,
                status: 'error',
              })
            )
          );
        }
      });
    setSelectedModelId(undefined);
  }, [
    deleteMainModel,
    deleteLoRAModel,
    model,
    setSelectedModelId,
    dispatch,
    t,
  ]);

  return (
    <Flex sx={{ gap: 2, alignItems: 'center', w: 'full' }}>
      <Flex
        as={InvButton}
        isChecked={isSelected}
        sx={{
          justifyContent: 'start',
          p: 2,
          borderRadius: 'base',
          w: 'full',
          alignItems: 'center',
          color: isSelected ? 'base.50' : 'base.100',
          bg: isSelected ? 'blue.600' : 'base.850',
          _hover: {
            color: isSelected ? 'base.50' : 'base.100',
            bg: isSelected ? 'blue.550' : 'base.700',
          },
        }}
        onClick={handleSelectModel}
      >
        <Flex gap={4} alignItems="center">
          <Badge minWidth={14} p={0.5} fontSize="sm" variant="solid">
            {
              MODEL_TYPE_SHORT_MAP[
                model.base_model as keyof typeof MODEL_TYPE_SHORT_MAP
              ]
            }
          </Badge>
          <InvTooltip label={model.description} placement="bottom">
            <InvText>{model.model_name}</InvText>
          </InvTooltip>
        </Flex>
      </Flex>
      <InvIconButton
        onClick={onOpen}
        icon={<DeleteIcon />}
        aria-label={t('modelManager.deleteConfig')}
        colorScheme="error"
      />
      <InvConfirmationAlertDialog
        isOpen={isOpen}
        onClose={onClose}
        title={t('modelManager.deleteModel')}
        acceptCallback={handleModelDelete}
        acceptButtonText={t('modelManager.delete')}
      >
        <Flex rowGap={4} flexDirection="column">
          <p style={{ fontWeight: 'bold' }}>{t('modelManager.deleteMsg1')}</p>
          <p>{t('modelManager.deleteMsg2')}</p>
        </Flex>
      </InvConfirmationAlertDialog>
    </Flex>
  );
}
