import { DeleteIcon, EditIcon } from '@chakra-ui/icons';
import {
  Box,
  Flex,
  Spacer,
  Text,
  Tooltip,
  useColorMode,
} from '@chakra-ui/react';

// import { deleteModel, requestModelChange } from 'app/socketio/actions';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIIconButton from 'common/components/IAIIconButton';
import { setOpenModel } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';
import { useDeleteMainModelsMutation } from 'services/api/endpoints/models';
import { BaseModelType } from 'services/api/types';
import { mode } from 'theme/util/mode';

type ModelListItemProps = {
  modelKey: string;
  name: string;
  description: string | undefined;
};

export default function ModelListItem(props: ModelListItemProps) {
  const { isProcessing, isConnected } = useAppSelector(
    (state: RootState) => state.system
  );

  const { colorMode } = useColorMode();

  const openModel = useAppSelector(
    (state: RootState) => state.system.openModel
  );

  const [deleteMainModel] = useDeleteMainModelsMutation();

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const { modelKey, name, description } = props;

  const openModelHandler = () => {
    dispatch(setOpenModel(modelKey));
  };

  const handleModelDelete = () => {
    const [base_model, _, model_name] = modelKey.split('/');
    deleteMainModel({
      base_model: base_model as BaseModelType,
      model_name: model_name,
    });
    dispatch(setOpenModel(null));
  };

  return (
    <Flex
      alignItems="center"
      p={2}
      borderRadius="base"
      sx={
        modelKey === openModel
          ? {
              bg: mode('accent.200', 'accent.600')(colorMode),
              _hover: {
                bg: mode('accent.200', 'accent.600')(colorMode),
              },
            }
          : {
              _hover: {
                bg: mode('base.100', 'base.800')(colorMode),
              },
            }
      }
    >
      <Box onClick={openModelHandler} cursor="pointer">
        <Tooltip label={description} hasArrow placement="bottom">
          <Text fontWeight="600">{name}</Text>
        </Tooltip>
      </Box>
      <Spacer onClick={openModelHandler} cursor="pointer" />
      <Flex gap={2} alignItems="center">
        <IAIIconButton
          icon={<EditIcon />}
          size="sm"
          onClick={openModelHandler}
          aria-label={t('accessibility.modifyConfig')}
          isDisabled={status === 'active' || isProcessing || !isConnected}
        />
        <IAIAlertDialog
          title={t('modelManager.deleteModel')}
          acceptCallback={handleModelDelete}
          acceptButtonText={t('modelManager.delete')}
          triggerComponent={
            <IAIIconButton
              icon={<DeleteIcon />}
              size="sm"
              aria-label={t('modelManager.deleteConfig')}
              isDisabled={status === 'active' || isProcessing || !isConnected}
              colorScheme="error"
            />
          }
        >
          <Flex rowGap={4} flexDirection="column">
            <p style={{ fontWeight: 'bold' }}>{t('modelManager.deleteMsg1')}</p>
            <p>{t('modelManager.deleteMsg2')}</p>
          </Flex>
        </IAIAlertDialog>
      </Flex>
    </Flex>
  );
}
