import { Divider, Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import ModelVariantSelect from 'features/modelManager/subpanels/shared/ModelVariantSelect';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { DiffusersModelConfigEntity } from 'services/api/endpoints/models';
import { useUpdateMainModelsMutation } from 'services/api/endpoints/models';
import type { DiffusersModelConfig } from 'services/api/types';

type DiffusersModelEditProps = {
  model: DiffusersModelConfigEntity;
};

export default function DiffusersModelEdit(props: DiffusersModelEditProps) {
  const { model } = props;

  const [updateMainModel, { isLoading }] = useUpdateMainModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const diffusersEditForm = useForm<DiffusersModelConfig>({
    initialValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'main',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: 'diffusers',
      vae: model.vae ? model.vae : '',
      variant: model.variant,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = useCallback(
    (values: DiffusersModelConfig) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };

      updateMainModel(responseBody)
        .unwrap()
        .then((payload) => {
          diffusersEditForm.setValues(payload as DiffusersModelConfig);
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelUpdated'),
                status: 'success',
              })
            )
          );
        })
        .catch((_) => {
          diffusersEditForm.reset();
          dispatch(
            addToast(
              makeToast({
                title: t('modelManager.modelUpdateFailed'),
                status: 'error',
              })
            )
          );
        });
    },
    [
      diffusersEditForm,
      dispatch,
      model.base_model,
      model.model_name,
      t,
      updateMainModel,
    ]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <InvText fontSize="lg" fontWeight="bold">
          {model.model_name}
        </InvText>
        <InvText fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} {t('modelManager.model')}
        </InvText>
      </Flex>
      <Divider />

      <form
        onSubmit={diffusersEditForm.onSubmit((values) =>
          editModelFormSubmitHandler(values)
        )}
      >
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <InvControl label={t('modelManager.name')}>
            <InvInput {...diffusersEditForm.getInputProps('model_name')} />
          </InvControl>
          <InvControl label={t('modelManager.description')}>
            <InvInput {...diffusersEditForm.getInputProps('description')} />
          </InvControl>
          <BaseModelSelect
            required
            {...diffusersEditForm.getInputProps('base_model')}
          />
          <ModelVariantSelect
            required
            {...diffusersEditForm.getInputProps('variant')}
          />
          <InvControl isRequired label={t('modelManager.modelLocation')}>
            <InvInput {...diffusersEditForm.getInputProps('path')} />
          </InvControl>
          <InvControl label={t('modelManager.vaeLocation')}>
            <InvInput {...diffusersEditForm.getInputProps('vae')} />
          </InvControl>
          <InvButton type="submit" isLoading={isLoading}>
            {t('modelManager.updateModel')}
          </InvButton>
        </Flex>
      </form>
    </Flex>
  );
}
