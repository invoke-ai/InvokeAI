import {
  Flex,
  ListItem,
  Radio,
  RadioGroup,
  Text,
  UnorderedList,
} from '@chakra-ui/react';
import { convertToDiffusers } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { setIsProcessing } from 'features/system/store/systemSlice';
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

  const [pathToConfig, setPathToConfig] = useState<string>('');
  const [modelType, setModelType] = useState<string>('1');

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const isConnected = useAppSelector(
    (state: RootState) => state.system.isConnected
  );

  // Need to manually handle local state reset because the component does not re-render.
  const stateReset = () => {
    setModelType('1');
    setPathToConfig('');
  };

  // Reset local state when model changes
  useEffect(() => {
    stateReset();
  }, [model]);

  // Handle local state reset when user cancels input
  const modelConvertCancelHandler = () => {
    stateReset();
  };

  const modelConvertHandler = () => {
    const modelConvertData = {
      name: model,
      model_type: modelType,
      custom_config:
        modelType === 'custom' && pathToConfig !== '' ? pathToConfig : null,
    };

    dispatch(setIsProcessing(true));
    dispatch(convertToDiffusers(modelConvertData));
    stateReset(); // Edge case: Cancel local state when model convert fails
  };

  return (
    <IAIAlertDialog
      title={`${t('modelmanager:convert')} ${model}`}
      acceptCallback={modelConvertHandler}
      cancelCallback={modelConvertCancelHandler}
      acceptButtonText={`${t('modelmanager:convert')}`}
      triggerComponent={
        <IAIButton
          size={'sm'}
          aria-label={t('modelmanager:convertToDiffusers')}
          isDisabled={
            retrievedModel.status === 'active' || isProcessing || !isConnected
          }
          className=" modal-close-btn"
          marginRight="2rem"
        >
          🧨 {t('modelmanager:convertToDiffusers')}
        </IAIButton>
      }
      motionPreset="slideInBottom"
    >
      <Flex flexDirection="column" rowGap={4}>
        <Text>{t('modelmanager:convertToDiffusersHelpText1')}</Text>
        <UnorderedList>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText2')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText3')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText4')}</ListItem>
          <ListItem>{t('modelmanager:convertToDiffusersHelpText5')}</ListItem>
        </UnorderedList>
        <Text>{t('modelmanager:convertToDiffusersHelpText6')}</Text>
        <RadioGroup
          value={modelType}
          onChange={(v) => setModelType(v)}
          defaultValue="1"
          name="model_type"
        >
          <Flex gap={4}>
            <Radio value="1">{t('modelmanager:v1')}</Radio>
            <Radio value="2">{t('modelmanager:v2')}</Radio>
            <Radio value="inpainting">{t('modelmanager:inpainting')}</Radio>
            <Radio value="custom">{t('modelmanager:custom')}</Radio>
          </Flex>
        </RadioGroup>
        {modelType === 'custom' && (
          <Flex flexDirection="column" rowGap={2}>
            <Text
              fontWeight="bold"
              fontSize="sm"
              color="var(--text-color-secondary)"
            >
              {t('modelmanager:pathToCustomConfig')}
            </Text>
            <IAIInput
              value={pathToConfig}
              onChange={(e) => {
                if (e.target.value !== '') setPathToConfig(e.target.value);
              }}
              width="25rem"
            />
          </Flex>
        )}
      </Flex>
    </IAIAlertDialog>
  );
}
