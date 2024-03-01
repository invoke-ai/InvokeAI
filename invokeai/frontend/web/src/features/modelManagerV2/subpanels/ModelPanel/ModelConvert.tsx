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
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useConvertMainModelsMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

interface ModelConvertProps {
  model: CheckpointModelConfig;
}

export const ModelConvert = (props: ModelConvertProps) => {
  const { model } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [convertModel, { isLoading }] = useConvertMainModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const modelConvertHandler = useCallback(() => {
    dispatch(
      addToast(
        makeToast({
          title: `${t('modelManager.convertingModelBegin')}: ${model.name}`,
          status: 'info',
        })
      )
    );

    convertModel(model.key)
      .unwrap()
      .then(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConverted')}: ${model.name}`,
              status: 'success',
            })
          )
        );
      })
      .catch(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConversionFailed')}: ${model.name}`,
              status: 'error',
            })
          )
        );
      });
  }, [convertModel, dispatch, model.key, model.name, t]);

  return (
    <>
      <Button
        onClick={onOpen}
        size="sm"
        aria-label={t('modelManager.convertToDiffusers')}
        className=" modal-close-btn"
        isLoading={isLoading}
      >
        🧨 {t('modelManager.convertToDiffusers')}
      </Button>
      <ConfirmationAlertDialog
        title={`${t('modelManager.convert')} ${model.name}`}
        acceptCallback={modelConvertHandler}
        acceptButtonText={`${t('modelManager.convert')}`}
        isOpen={isOpen}
        onClose={onClose}
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
};
