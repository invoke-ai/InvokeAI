import { Divider, Flex } from '@chakra-ui/react';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { InvButton } from 'common/components/InvButton/InvButton';
import { InvControl } from 'common/components/InvControl/InvControl';
import { InvInput } from 'common/components/InvInput/InvInput';
import { InvText } from 'common/components/InvText/wrapper';
import BaseModelSelect from 'features/modelManager/subpanels/shared/BaseModelSelect';
import {
  LORA_MODEL_FORMAT_MAP,
  MODEL_TYPE_MAP,
} from 'features/parameters/types/constants';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { LoRAModelConfigEntity } from 'services/api/endpoints/models';
import { useUpdateLoRAModelsMutation } from 'services/api/endpoints/models';
import type { LoRAModelConfig } from 'services/api/types';

type LoRAModelEditProps = {
  model: LoRAModelConfigEntity;
};

const LoRAModelEdit = (props: LoRAModelEditProps) => {
  const { model } = props;

  const [updateLoRAModel, { isLoading }] = useUpdateLoRAModelsMutation();

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const loraEditForm = useForm<LoRAModelConfig>({
    initialValues: {
      model_name: model.model_name ? model.model_name : '',
      base_model: model.base_model,
      model_type: 'lora',
      path: model.path ? model.path : '',
      description: model.description ? model.description : '',
      model_format: model.model_format,
    },
    validate: {
      path: (value) =>
        value.trim().length === 0 ? 'Must provide a path' : null,
    },
  });

  const editModelFormSubmitHandler = useCallback(
    (values: LoRAModelConfig) => {
      const responseBody = {
        base_model: model.base_model,
        model_name: model.model_name,
        body: values,
      };

      updateLoRAModel(responseBody)
        .unwrap()
        .then((payload) => {
          loraEditForm.setValues(payload as LoRAModelConfig);
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
          loraEditForm.reset();
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
      dispatch,
      loraEditForm,
      model.base_model,
      model.model_name,
      t,
      updateLoRAModel,
    ]
  );

  return (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex flexDirection="column">
        <InvText fontSize="lg" fontWeight="bold">
          {model.model_name}
        </InvText>
        <InvText fontSize="sm" color="base.400">
          {MODEL_TYPE_MAP[model.base_model]} {t('modelManager.model')} ⋅{' '}
          {LORA_MODEL_FORMAT_MAP[model.model_format]} {t('common.format')}
        </InvText>
      </Flex>
      <Divider />

      <form
        onSubmit={loraEditForm.onSubmit((values) =>
          editModelFormSubmitHandler(values)
        )}
      >
        <Flex flexDirection="column" overflowY="scroll" gap={4}>
          <InvControl label={t('modelManager.name')}>
            <InvInput {...loraEditForm.getInputProps('model_name')} />
          </InvControl>
          <InvControl label={t('modelManager.description')}>
            <InvInput {...loraEditForm.getInputProps('description')} />
          </InvControl>
          <BaseModelSelect {...loraEditForm.getInputProps('base_model')} />
          <InvControl label={t('modelManager.modelLocation')}>
            <InvInput {...loraEditForm.getInputProps('path')} />
          </InvControl>
          <InvButton type="submit" isLoading={isLoading}>
            {t('modelManager.updateModel')}
          </InvButton>
        </Flex>
      </form>
    </Flex>
  );
};

export default memo(LoRAModelEdit);
