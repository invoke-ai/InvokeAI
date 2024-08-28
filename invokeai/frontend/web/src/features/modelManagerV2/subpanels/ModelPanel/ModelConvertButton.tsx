import {
  Button,
  ConfirmationAlertDialog,
  Divider,
  Flex,
  ListItem,
  Text,
  UnorderedList,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useConvertModelMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

interface ModelConvertProps {
  modelConfig: CheckpointModelConfig;
}

export const ModelConvertButton = memo(({ modelConfig }: ModelConvertProps) => {
  const { t } = useTranslation();
  const [convertModel, { isLoading }] = useConvertModelMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const modelConvertHandler = useCallback(() => {
    if (!modelConfig || isLoading) {
      return;
    }

    const toastId = `CONVERTING_MODEL_${modelConfig.key}`;
    toast({
      id: toastId,
      title: `${t('modelManager.convertingModelBegin')}: ${modelConfig.name}`,
      status: 'info',
    });

    convertModel(modelConfig.key)
      .unwrap()
      .then(() => {
        toast({ id: toastId, title: `${t('modelManager.modelConverted')}: ${modelConfig.name}`, status: 'success' });
      })
      .catch(() => {
        toast({
          id: toastId,
          title: `${t('modelManager.modelConversionFailed')}: ${modelConfig.name}`,
          status: 'error',
        });
      });
  }, [modelConfig, isLoading, t, convertModel]);

  return (
    <>
      <Button
        onClick={onOpen}
        size="sm"
        aria-label={t('modelManager.convertToDiffusers')}
        className=" modal-close-btn"
        isLoading={isLoading}
        flexShrink={0}
      >
        🧨 {t('modelManager.convert')}
      </Button>
      <ConfirmationAlertDialog
        title={`${t('modelManager.convert')} ${modelConfig.name}`}
        acceptCallback={modelConvertHandler}
        acceptButtonText={`${t('modelManager.convert')}`}
        isOpen={isOpen}
        onClose={onClose}
        useInert={false}
      >
        <Flex flexDirection="column" rowGap={4}>
          <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText1')}</Text>
          <UnorderedList>
            <ListItem>
              <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText2')}</Text>
            </ListItem>
            <ListItem>
              <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText3')}</Text>
            </ListItem>
            <ListItem>
              <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText4')}</Text>
            </ListItem>
            <ListItem>
              <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText5')}</Text>
            </ListItem>
          </UnorderedList>
          <Divider />
          <Text fontSize="md">{t('modelManager.convertToDiffusersHelpText6')}</Text>
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
});

ModelConvertButton.displayName = 'ModelConvertButton';
