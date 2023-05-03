import { DeleteIcon, EditIcon } from '@chakra-ui/icons';
import { Box, Button, Flex, Spacer, Text, Tooltip } from '@chakra-ui/react';
import { ModelStatus } from 'app/types/invokeai';
// import { deleteModel, requestModelChange } from 'app/socketio/actions';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIIconButton from 'common/components/IAIIconButton';
import { setOpenModel } from 'features/system/store/systemSlice';
import { useTranslation } from 'react-i18next';

type ModelListItemProps = {
  name: string;
  status: ModelStatus;
  description: string;
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

  const { name, status, description } = props;

  const handleChangeModel = () => {
    dispatch(requestModelChange(name));
  };

  const openModelHandler = () => {
    dispatch(setOpenModel(name));
  };

  const handleModelDelete = () => {
    dispatch(deleteModel(name));
    dispatch(setOpenModel(null));
  };

  const statusTextColor = () => {
    switch (status) {
      case 'active':
        return 'ok.500';
      case 'cached':
        return 'warning.500';
      case 'not loaded':
        return 'inherit';
    }
  };

  return (
    <Flex
      alignItems="center"
      p={2}
      borderRadius="base"
      sx={
        name === openModel
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
        <Text color={statusTextColor()}>{status}</Text>
        <Button
          size="sm"
          onClick={handleChangeModel}
          isDisabled={status === 'active' || isProcessing || !isConnected}
        >
          {t('modelManager.load')}
        </Button>

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
