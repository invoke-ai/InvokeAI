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
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useConvertModelMutation, useGetModelConfigQuery } from 'services/api/endpoints/models';

interface ModelConvertProps {
  modelKey: string | null;
}

export const ModelConvertButton = (props: ModelConvertProps) => {
  const { modelKey } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const { data } = useGetModelConfigQuery(modelKey ?? skipToken);
  const [convertModel, { isLoading }] = useConvertModelMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  const modelConvertHandler = useCallback(() => {
    if (!data || isLoading) {
      return;
    }

    dispatch(
      addToast(
        makeToast({
          title: `${t('modelManager.convertingModelBegin')}: ${data?.name}`,
          status: 'info',
        })
      )
    );

    convertModel(data?.key)
      .unwrap()
      .then(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConverted')}: ${data?.name}`,
              status: 'success',
            })
          )
        );
      })
      .catch(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConversionFailed')}: ${data?.name}`,
              status: 'error',
            })
          )
        );
      });
  }, [data, isLoading, dispatch, t, convertModel]);

  if (data?.format !== 'checkpoint') {
    return;
  }

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
        title={`${t('modelManager.convert')} ${data?.name}`}
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
