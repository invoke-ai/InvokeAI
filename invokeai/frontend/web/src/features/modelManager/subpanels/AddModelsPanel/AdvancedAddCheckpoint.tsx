import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvCheckbox } from 'common/components/InvCheckbox/wrapper';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import CheckpointConfigsSelect from 'features/modelManager/subpanels/shared/CheckpointConfigsSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties, FocusEventHandler } from 'react';
import { useCallback, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import type { CheckpointModelConfig } from 'services/api/types';

import { getModelName } from './util';

type AdvancedAddCheckpointProps = {
  model_path?: string;
};

export default function AdvancedAddCheckpoint(
  props: AdvancedAddCheckpointProps
) {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const advancedAddCheckpointForm = useForm<CheckpointModelConfig>({
    initialValues: {
      model_name: model_path ? getModelName(model_path) : '',
      base_model: 'sd-1',
      model_type: 'main',
      path: model_path ? model_path : '',
      description: '',
      model_format: 'checkpoint',
      error: undefined,
      vae: '',
      variant: 'normal',
      config: 'configs\\stable-diffusion\\v1-inference.yaml',
    },
  });

  const [addMainModel] = useAddMainModelsMutation();

  const [useCustomConfig, setUseCustomConfig] = useState<boolean>(false);

  const advancedAddCheckpointFormHandler = (values: CheckpointModelConfig) => {
    addMainModel({
      body: values,
    })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('modelManager.modelAdded', {
                modelName: values.model_name,
              }),
              status: 'success',
            })
          )
        );
        advancedAddCheckpointForm.reset();

        // Close Advanced Panel in Scan Models tab
        if (model_path) {
          dispatch(setAdvancedAddScanModel(null));
        }
      })
      .catch((error) => {
        if (error) {
          dispatch(
            addToast(
              makeToast({
                title: t('toast.modelAddFailed'),
                status: 'error',
              })
            )
          );
        }
      });
  };

  const handleBlurModelLocation: FocusEventHandler<HTMLInputElement> =
    useCallback(
      (e) => {
        if (advancedAddCheckpointForm.values['model_name'] === '') {
          const modelName = getModelName(e.currentTarget.value);
          if (modelName) {
            advancedAddCheckpointForm.setFieldValue(
              'model_name',
              modelName as string
            );
          }
        }
      },
      [advancedAddCheckpointForm]
    );

  const handleChangeUseCustomConfig = useCallback(
    () => setUseCustomConfig((prev) => !prev),
    []
  );

  return (
    <form
      onSubmit={advancedAddCheckpointForm.onSubmit((v) =>
        advancedAddCheckpointFormHandler(v)
      )}
      style={formStyles}
    >
      <Flex flexDirection="column" gap={2}>
        <InvControl label={t('modelManager.model')} isRequired>
          <InvInput
            {...advancedAddCheckpointForm.getInputProps('model_name')}
          />
        </InvControl>
        <InvControl label={t('modelManager.baseModel')}>
          <BaseModelSelect
            {...advancedAddCheckpointForm.getInputProps('base_model')}
          />
        </InvControl>
        <InvControl label={t('modelManager.modelLocation')} isRequired>
          <InvInput
            {...advancedAddCheckpointForm.getInputProps('path')}
            onBlur={handleBlurModelLocation}
          />
        </InvControl>
        <InvControl label={t('modelManager.description')}>
          <InvInput
            {...advancedAddCheckpointForm.getInputProps('description')}
          />
        </InvControl>
        <InvControl label={t('modelManager.vaeLocation')}>
          <InvInput {...advancedAddCheckpointForm.getInputProps('vae')} />
        </InvControl>
        <InvControl label={t('modelManager.variant')}>
          <ModelVariantSelect
            {...advancedAddCheckpointForm.getInputProps('variant')}
          />
        </InvControl>
        <Flex flexDirection="column" width="100%" gap={2}>
          {!useCustomConfig ? (
            <CheckpointConfigsSelect
              required
              {...advancedAddCheckpointForm.getInputProps('config')}
            />
          ) : (
            <InvControl
              label={t('modelManager.customConfigFileLocation')}
              isRequired
            >
              <InvInput
                {...advancedAddCheckpointForm.getInputProps('config')}
              />
            </InvControl>
          )}
          <InvControl label={t('modelManager.useCustomConfig')}>
            <InvCheckbox
              isChecked={useCustomConfig}
              onChange={handleChangeUseCustomConfig}
            />
          </InvControl>
          <InvButton mt={2} type="submit">
            {t('modelManager.addModel')}
          </InvButton>
        </Flex>
      </Flex>
    </form>
  );
}

const formStyles: CSSProperties = {
  width: '100%',
};
