import {
  Flex,
  ListItem,
  Radio,
  RadioGroup,
  UnorderedList,
  useDisclosure,
} from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvConfirmationAlertDialog } from 'common/components/InvConfirmationAlertDialog/InvConfirmationAlertDialog';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { ChangeEvent } from 'react';
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useConvertMainModelsMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

interface ModelConvertProps {
  model: CheckpointModelConfig;
}

type SaveLocation = 'InvokeAIRoot' | 'Custom';

export default function ModelConvert(props: ModelConvertProps) {
  const { model } = props;
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const [convertModel, { isLoading }] = useConvertMainModelsMutation();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [saveLocation, setSaveLocation] =
    useState<SaveLocation>('InvokeAIRoot');
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
  const handleChangeCustomSaveLocation = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      setCustomSaveLocation(e.target.value);
    },
    []
  );

  const modelConvertHandler = useCallback(() => {
    const queryArg = {
      base_model: model.base_model,
      model_name: model.model_name,
      convert_dest_directory:
        saveLocation === 'Custom' ? customSaveLocation : undefined,
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
          title: `${t('modelManager.convertingModelBegin')}: ${
            model.model_name
          }`,
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
              title: `${t('modelManager.modelConversionFailed')}: ${
                model.model_name
              }`,
              status: 'error',
            })
          )
        );
      });
  }, [
    convertModel,
    customSaveLocation,
    dispatch,
    model.base_model,
    model.model_name,
    saveLocation,
    t,
  ]);

  return (
    <>
      <InvButton
        onClick={onOpen}
        size="sm"
        aria-label={t('modelManager.convertToDiffusers')}
        className=" modal-close-btn"
        isLoading={isLoading}
      >
        🧨 {t('modelManager.convertToDiffusers')}
      </InvButton>
      <InvConfirmationAlertDialog
        title={`${t('modelManager.convert')} ${model.model_name}`}
        acceptCallback={modelConvertHandler}
        cancelCallback={modelConvertCancelHandler}
        acceptButtonText={`${t('modelManager.convert')}`}
        isOpen={isOpen}
        onClose={onClose}
      >
        <Flex flexDirection="column" rowGap={4}>
          <InvText>{t('modelManager.convertToDiffusersHelpText1')}</InvText>
          <UnorderedList>
            <ListItem>{t('modelManager.convertToDiffusersHelpText2')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText3')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText4')}</ListItem>
            <ListItem>{t('modelManager.convertToDiffusersHelpText5')}</ListItem>
          </UnorderedList>
          <InvText>{t('modelManager.convertToDiffusersHelpText6')}</InvText>
        </Flex>

        <Flex flexDir="column" gap={2}>
          <Flex marginTop={4} flexDir="column" gap={2}>
            <InvText fontWeight="semibold">
              {t('modelManager.convertToDiffusersSaveLocation')}
            </InvText>
            <RadioGroup
              value={saveLocation}
              onChange={handleChangeSaveLocation}
            >
              <Flex gap={4}>
                <Radio value="InvokeAIRoot">
                  <InvTooltip label="Save converted model in the InvokeAI root folder">
                    {t('modelManager.invokeRoot')}
                  </InvTooltip>
                </Radio>
                <Radio value="Custom">
                  <InvTooltip label="Save converted model in a custom folder">
                    {t('modelManager.custom')}
                  </InvTooltip>
                </Radio>
              </Flex>
            </RadioGroup>
          </Flex>
          {saveLocation === 'Custom' && (
            <InvControl label={t('modelManager.customSaveLocation')}>
              <InvInput
                width="full"
                value={customSaveLocation}
                onChange={handleChangeCustomSaveLocation}
              />
            </InvControl>
          )}
        </Flex>
      </InvConfirmationAlertDialog>
    </>
  );
}
