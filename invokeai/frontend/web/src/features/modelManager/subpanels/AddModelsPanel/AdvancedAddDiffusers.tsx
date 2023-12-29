import { Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { setAdvancedAddScanModel } from 'features/modelManager/store/modelManagerSlice';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties, FocusEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useAddMainModelsMutation } from 'services/api/endpoints/models';
import type { DiffusersModelConfig } from 'services/api/types';

import { getModelName } from './util';

type AdvancedAddDiffusersProps = {
  model_path?: string;
};

const AdvancedAddDiffusers = (props: AdvancedAddDiffusersProps) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const { model_path } = props;

  const [addMainModel] = useAddMainModelsMutation();

  const advancedAddDiffusersForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: model_path ? getModelName(model_path, false) : '',
      base_model: 'sd-1',
      model_type: 'main',
      path: model_path ? model_path : '',
      description: '',
      model_format: 'diffusers',
      error: undefined,
      vae: '',
      variant: 'normal',
    },
  });

  const advancedAddDiffusersFormHandler = (values: DiffusersModelConfig) => {
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
        advancedAddDiffusersForm.reset();
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
        if (advancedAddDiffusersForm.values['model_name'] === '') {
          const modelName = getModelName(e.currentTarget.value, false);
          if (modelName) {
            advancedAddDiffusersForm.setFieldValue(
              'model_name',
              modelName as string
            );
          }
        }
      },
      [advancedAddDiffusersForm]
    );

  return (
    <form
      onSubmit={advancedAddDiffusersForm.onSubmit((v) =>
        advancedAddDiffusersFormHandler(v)
      )}
      style={formStyles}
    >
      <Flex flexDirection="column" gap={2}>
        <InvControl isRequired label={t('modelManager.model')}>
          <InvInput {...advancedAddDiffusersForm.getInputProps('model_name')} />
        </InvControl>
        <InvControl label={t('modelManager.baseModel')}>
          <BaseModelSelect
            {...advancedAddDiffusersForm.getInputProps('base_model')}
          />
        </InvControl>
        <InvControl isRequired label={t('modelManager.modelLocation')}>
          <InvInput
            placeholder={t('modelManager.modelLocationValidationMsg')}
            {...advancedAddDiffusersForm.getInputProps('path')}
            onBlur={handleBlurModelLocation}
          />
        </InvControl>
        <InvControl label={t('modelManager.description')}>
          <InvInput
            {...advancedAddDiffusersForm.getInputProps('description')}
          />
        </InvControl>
        <InvControl label={t('modelManager.vaeLocation')}>
          <InvInput {...advancedAddDiffusersForm.getInputProps('vae')} />
        </InvControl>
        <InvControl label={t('modelManager.variant')}>
          <ModelVariantSelect
            {...advancedAddDiffusersForm.getInputProps('variant')}
          />
        </InvControl>

        <InvButton mt={2} type="submit">
          {t('modelManager.addModel')}
        </InvButton>
      </Flex>
    </form>
  );
};

const formStyles: CSSProperties = {
  width: '100%',
};

export default memo(AdvancedAddDiffusers);
