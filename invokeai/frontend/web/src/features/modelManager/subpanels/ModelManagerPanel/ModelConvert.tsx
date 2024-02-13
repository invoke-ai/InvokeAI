import {
  Button,
  ConfirmationAlertDialog,
  Flex,
  FormControl,
  FormLabel,
  Input,
  ListItem,
  Radio,
  RadioGroup,
  Text,
  Tooltip,
  UnorderedList,
  useDisclosure,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { ChangeEvent } from 'react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useConvertMainModelsMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

interface ModelConvertProps {
  model: CheckpointModelConfig;
}

type SaveLocation = 'InvokeAIRoot' | 'Custom';

const ModelConvert = (props: ModelConvertProps) => {
  const { model } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [convertModel, { isLoading }] = useConvertMainModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [saveLocation, setSaveLocation] = useState<SaveLocation>('InvokeAIRoot');
  const [customSaveLocation, setCustomSaveLocation] = useState<string>('');

  useEffect(() => {
    setSaveLocation('InvokeAIRoot');
  }, [model]);

  const modelConvertCancelHandler = useCallback(() => {
    setSaveLocation('InvokeAIRoot');
  }, []);

  const handleChangeSaveLocation = useCallback((v: string) => {
    setSaveLocation(v as SaveLocation);
  }, []);
  const handleChangeCustomSaveLocation = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    setCustomSaveLocation(e.target.value);
  }, []);

  const modelConvertHandler = useCallback(() => {
    const queryArg = {
      base_model: model.base_model,
      model_name: model.model_name,
      convert_dest_directory: saveLocation === 'Custom' ? customSaveLocation : undefined,
    };

    if (saveLocation === 'Custom' && customSaveLocation === '') {
      dispatch(
        addToast(
          makeToast({
            title: t('modelManager.noCustomLocationProvided'),
            status: 'error',
          })
        )
      );
      return;
    }

    dispatch(
      addToast(
        makeToast({
          title: `${t('modelManager.convertingModelBegin')}: ${model.model_name}`,
          status: 'info',
        })
      )
    );

    convertModel(queryArg)
      .unwrap()
      .then(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConverted')}: ${model.model_name}`,
              status: 'success',
            })
          )
        );
      })
      .catch(() => {
        dispatch(
          addToast(
            makeToast({
              title: `${t('modelManager.modelConversionFailed')}: ${model.model_name}`,
              status: 'error',
            })
          )
        );
      });
  }, [convertModel, customSaveLocation, dispatch, model.base_model, model.model_name, saveLocation, t]);

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
        title={`${t('modelManager.convert')} ${model.model_name}`}
        acceptCallback={modelConvertHandler}
        cancelCallback={modelConvertCancelHandler}
        acceptButtonText={`${t('modelManager.convert')}`}
        isOpen={isOpen}
        onClose={onClose}
      >
        <Flex flexDirection="column" rowGap={4}>
          <Text>{t('modelManager.convertToDiffusersHelpText1')}</Text>
          <UnorderedList>
            <ListItem>{t('modelManager.convertToDiffusersHelpText2')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText3')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText4')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText5')}</ListItem>
          </UnorderedList>
          <Text>{t('modelManager.convertToDiffusersHelpText6')}</Text>
        </Flex>

        <Flex flexDir="column" gap={2}>
          <Flex marginTop={4} flexDir="column" gap={2}>
            <Text fontWeight="semibold">{t('modelManager.convertToDiffusersSaveLocation')}</Text>
            <RadioGroup value={saveLocation} onChange={handleChangeSaveLocation}>
              <Flex gap={4}>
                <Radio value="InvokeAIRoot">
                  <Tooltip label="Save converted model in the InvokeAI root folder">
                    {t('modelManager.invokeRoot')}
                  </Tooltip>
                </Radio>
                <Radio value="Custom">
                  <Tooltip label="Save converted model in a custom folder">{t('modelManager.custom')}</Tooltip>
                </Radio>
              </Flex>
            </RadioGroup>
          </Flex>
          {saveLocation === 'Custom' && (
            <FormControl>
              <FormLabel>{t('modelManager.customSaveLocation')}</FormLabel>
              <Input width="full" value={customSaveLocation} onChange={handleChangeCustomSaveLocation} />
            </FormControl>
          )}
        </Flex>
      </ConfirmationAlertDialog>
    </>
  );
};

export default memo(ModelConvert);
