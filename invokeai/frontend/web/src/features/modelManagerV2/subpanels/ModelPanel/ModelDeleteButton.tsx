import { Button, ConfirmationAlertDialog, Flex, IconButton, Text, useDisclosure } from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { toast } from 'features/toast/toast';
import { memo, type MouseEvent, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';
import { useDeleteModelsMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

type Props = {
  showLabel?: boolean;
  modelConfig: AnyModelConfig;
};

export const ModelDeleteButton = memo(({ showLabel = true, modelConfig }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const log = logger('models');

  const [deleteModel] = useDeleteModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const onClickDeleteButton = useCallback(
    (e: MouseEvent<HTMLButtonElement>) => {
      e.stopPropagation();
      onOpen();
    },
    [onOpen]
  );

  const handleModelDelete = useCallback(() => {
    deleteModel({ key: modelConfig.key })
      .unwrap()
      .then(() => {
        dispatch(setSelectedModelKey(null));
        toast({
          id: 'MODEL_DELETED',
          title: `${t('modelManager.modelDeleted')}: ${modelConfig.name}`,
          status: 'success',
        });
      })
      .catch((error) => {
        log.error('Error deleting model', error);
        toast({
          id: 'MODEL_DELETE_FAILED',
          title: `${t('modelManager.modelDeleteFailed')}: ${modelConfig.name}`,
          status: 'error',
        });
      });
  }, [deleteModel, modelConfig.key, modelConfig.name, dispatch, t, log]);

  return (
    <>
      {showLabel ? (
        <Button
          className="delete-button"
          size="sm"
          leftIcon={<PiTrashSimpleBold />}
          colorScheme="error"
          onClick={onClickDeleteButton}
          flexShrink={0}
        >
          {t('modelManager.delete')}
        </Button>
      ) : (
        <IconButton
          className="delete-button"
          onClick={onClickDeleteButton}
          icon={<PiTrashSimpleBold size={16} />}
          aria-label={t('modelManager.deleteConfig')}
          colorScheme="error"
        />
      )}

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
    </>
  );
});

ModelDeleteButton.displayName = 'ModelDeleteButton';
