import {
  Flex,
  ListItem,
  Radio,
  RadioGroup,
  Text,
  Tooltip,
  UnorderedList,
} from '@chakra-ui/react';
import { makeToast } from 'features/system/util/makeToast';
// import { convertToDiffusers } from 'app/socketio/actions';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIAlertDialog from 'common/components/IAIAlertDialog';
import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { addToast } from 'features/system/store/systemSlice';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useConvertMainModelsMutation } from 'services/api/endpoints/models';
import { CheckpointModelConfig } from 'services/api/types';

interface ModelConvertProps {
  model: CheckpointModelConfig;
}

type SaveLocation = 'InvokeAIRoot' | 'Custom';

export default function ModelConvert(props: ModelConvertProps) {
  const { model } = props;

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [convertModel, { isLoading }] = useConvertMainModelsMutation();

  const [saveLocation, setSaveLocation] =
    useState<SaveLocation>('InvokeAIRoot');
  const [customSaveLocation, setCustomSaveLocation] = useState<string>('');

  useEffect(() => {
    setSaveLocation('InvokeAIRoot');
  }, [model]);

  const modelConvertCancelHandler = () => {
    setSaveLocation('InvokeAIRoot');
  };

  const modelConvertHandler = () => {
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
  };

  return (
    <IAIAlertDialog
      title={`${t('modelManager.convert')} ${model.model_name}`}
      acceptCallback={modelConvertHandler}
      cancelCallback={modelConvertCancelHandler}
      acceptButtonText={`${t('modelManager.convert')}`}
      triggerComponent={
        <IAIButton
          size="sm"
          aria-label={t('modelManager.convertToDiffusers')}
          className=" modal-close-btn"
          isLoading={isLoading}
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

      <Flex flexDir="column" gap={2}>
        <Flex marginTop={4} flexDir="column" gap={2}>
          <Text fontWeight="600">
            {t('modelManager.convertToDiffusersSaveLocation')}
          </Text>
          <RadioGroup
            value={saveLocation}
            onChange={(v) => setSaveLocation(v as SaveLocation)}
          >
            <Flex gap={4}>
              <Radio value="InvokeAIRoot">
                <Tooltip label="Save converted model in the InvokeAI root folder">
                  {t('modelManager.invokeRoot')}
                </Tooltip>
              </Radio>
              <Radio value="Custom">
                <Tooltip label="Save converted model in a custom folder">
                  {t('modelManager.custom')}
                </Tooltip>
              </Radio>
            </Flex>
          </RadioGroup>
        </Flex>
        {saveLocation === 'Custom' && (
          <Flex flexDirection="column" rowGap={2}>
            <Text fontWeight="500" fontSize="sm" variant="subtext">
              {t('modelManager.customSaveLocation')}
            </Text>
            <IAIInput
              value={customSaveLocation}
              onChange={(e) => {
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
