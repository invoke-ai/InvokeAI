import { DeleteIcon, EditIcon } from '@chakra-ui/icons';
import { Box, Flex, Spacer, Text, Tooltip } from '@chakra-ui/react';

// import { deleteModel, requestModelChange } from 'app/socketio/actions';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIIconButton from 'common/components/IAIIconButton';
import { setOpenModel } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';

type ModelListItemProps = {
  modelKey: string;
  name: string;
  description: string | undefined;
};

export default function ModelListItem(props: ModelListItemProps) {
  const { isProcessing, isConnected } = useAppSelector(
    (state: RootState) => state.system
  );

  const openModel = useAppSelector(
    (state: RootState) => state.system.openModel
  );

  const { t } = useTranslation();

  const dispatch = useAppDispatch();

  const { modelKey, name, description } = props;

  const openModelHandler = () => {
    dispatch(setOpenModel(modelKey));
  };

  const handleModelDelete = () => {
    dispatch(deleteModel(modelKey));
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
              bg: 'accent.750',
              _hover: {
                bg: 'accent.750',
              },
            }
          : {
              _hover: {
                bg: 'base.750',
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
