import {
  Flex,
  ListItem,
  Radio,
  RadioGroup,
  Text,
  UnorderedList,
  Tooltip,
} from '@chakra-ui/react';
// import { convertToDiffusers } from 'app/socketio/actions';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

interface ModelConvertProps {
  model: string;
}

export default function ModelConvert(props: ModelConvertProps) {
  const { model } = props;

  const model_list = useAppSelector(
    (state: RootState) => state.system.model_list
  );

  const retrievedModel = model_list[model];

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const isConnected = useAppSelector(
    (state: RootState) => state.system.isConnected
  );

  const [saveLocation, setSaveLocation] = useState<string>('same');
  const [customSaveLocation, setCustomSaveLocation] = useState<string>('');

  useEffect(() => {
    setSaveLocation('same');
  }, [model]);

  const modelConvertCancelHandler = () => {
    setSaveLocation('same');
  };

  const modelConvertHandler = () => {
    const modelToConvert = {
      model_name: model,
      save_location: saveLocation,
      custom_location:
        saveLocation === 'custom' && customSaveLocation !== ''
          ? customSaveLocation
          : null,
    };
    dispatch(convertToDiffusers(modelToConvert));
  };

  return (
    <IAIAlertDialog
      title={`${t('modelManager.convert')} ${model}`}
      acceptCallback={modelConvertHandler}
      cancelCallback={modelConvertCancelHandler}
      acceptButtonText={`${t('modelManager.convert')}`}
      triggerComponent={
        <IAIButton
          size={'sm'}
          aria-label={t('modelManager.convertToDiffusers')}
          isDisabled={
            retrievedModel.status === 'active' || isProcessing || !isConnected
          }
          className=" modal-close-btn"
          marginInlineEnd={8}
        >
          🧨 {t('modelManager.convertToDiffusers')}
        </IAIButton>
      }
      motionPreset="slideInBottom"
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

      <Flex flexDir="column" gap={4}>
        <Flex marginTop={4} flexDir="column" gap={2}>
          <Text fontWeight="600">
            {t('modelManager.convertToDiffusersSaveLocation')}
          </Text>
          <RadioGroup value={saveLocation} onChange={(v) => setSaveLocation(v)}>
            <Flex gap={4}>
              <Radio value="same">
                <Tooltip label="Save converted model in the same folder">
                  {t('modelManager.sameFolder')}
                </Tooltip>
              </Radio>

              <Radio value="root">
                <Tooltip label="Save converted model in the InvokeAI root folder">
                  {t('modelManager.invokeRoot')}
                </Tooltip>
              </Radio>

              <Radio value="custom">
                <Tooltip label="Save converted model in a custom folder">
                  {t('modelManager.custom')}
                </Tooltip>
              </Radio>
            </Flex>
          </RadioGroup>
        </Flex>

        {saveLocation === 'custom' && (
          <Flex flexDirection="column" rowGap={2}>
            <Text fontWeight="500" fontSize="sm" variant="subtext">
              {t('modelManager.customSaveLocation')}
            </Text>
            <IAIInput
              value={customSaveLocation}
              onChange={(e) => {
                if (e.target.value !== '')
                  setCustomSaveLocation(e.target.value);
              }}
              width="full"
            />
          </Flex>
        )}
      </Flex>
    </IAIAlertDialog>
  );
}
