import { DeleteIcon } from '@chakra-ui/icons';
import {
  Button,
  Flex,
  IconButton,
  Spacer,
  Text,
  Tooltip,
} from '@chakra-ui/react';
import { ModelStatus } from 'app/invokeai';
import { deleteModel, requestModelChange } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import React from 'react';
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

  const { t } = useTranslation();

  const dispatch = useAppDispatch();
  const { name, status, description } = props;

  const handleChangeModel = () => {
    dispatch(requestModelChange(name));
  };

  const handleModelDelete = () => {
    dispatch(deleteModel(name));
  };

  const statusTextColor = () => {
    switch (status) {
      case 'active':
        return 'var(--status-good-color)';
      case 'cached':
        return 'var(--status-working-color)';
      case 'not loaded':
        return 'var(--text-color-secondary)';
    }
  };

  return (
    <Flex alignItems={'center'}>
      <Tooltip label={description} hasArrow placement="bottom">
        <Text fontWeight={'bold'}>{name}</Text>
      </Tooltip>
      <Spacer />

      <Flex gap={4} alignItems="center">
        <Text color={statusTextColor()}>{status}</Text>

        <Button
          size={'sm'}
          onClick={handleChangeModel}
          isDisabled={status === 'active' || isProcessing || !isConnected}
          className="modal-close-btn"
        >
          {t('modelmanager:load')}
        </Button>
        <IAIAlertDialog
          title={t('modelmanager:deleteModel')}
          acceptCallback={handleModelDelete}
          acceptButtonText={t('modelmanager:delete')}
          triggerComponent={
            <IconButton
              icon={<DeleteIcon />}
              size={'sm'}
              aria-label={t('modelmanager:deleteConfig')}
              isDisabled={status === 'active' || isProcessing || !isConnected}
              className=" modal-close-btn"
            />
          }
        >
          <Flex rowGap={'1rem'} flexDirection="column">
            <p style={{ fontWeight: 'bold' }}>{t('modelmanager:deleteMsg1')}</p>
            <p style={{ color: 'var(--text-color-secondary' }}>
              {t('modelmanager:deleteMsg2')}
            </p>
          </Flex>
        </IAIAlertDialog>
      </Flex>
    </Flex>
  );
}
