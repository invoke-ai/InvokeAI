import { Flex, ListItem, Text, UnorderedList } from '@chakra-ui/react';
import { convertToDiffusers } from 'app/socketio/actions';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
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

  const [isInpainting, setIsInpainting] = useState<boolean>(false);
  const [customConfig, setIsCustomConfig] = useState<boolean>(false);
  const [pathToConfig, setPathToConfig] = useState<string>('');

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
    setIsInpainting(false);
    setIsCustomConfig(false);
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
      is_inpainting: isInpainting,
      custom_config: customConfig && pathToConfig !== '' ? pathToConfig : null,
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
        <Flex flexDir="column" gap={4}>
          <Flex gap={4}>
            <IAICheckbox
              checked={isInpainting}
              onChange={() => {
                setIsInpainting(!isInpainting);
                setIsCustomConfig(false);
              }}
              label={t('modelmanager:inpaintingModel')}
              isDisabled={customConfig}
            />
            <IAICheckbox
              checked={customConfig}
              onChange={() => {
                setIsCustomConfig(!customConfig);
                setIsInpainting(false);
              }}
              label={t('modelmanager:customConfig')}
              isDisabled={isInpainting}
            />
          </Flex>
          {customConfig && (
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
      </Flex>
    </IAIAlertDialog>
  );
}
