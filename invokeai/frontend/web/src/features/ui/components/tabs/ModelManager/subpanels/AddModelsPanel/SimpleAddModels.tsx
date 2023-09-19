import { Flex } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

import { SelectItem } from '@mantine/core';
import { useForm } from '@mantine/form';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useImportMainModelsMutation } from 'services/api/endpoints/models';

const predictionSelectData: SelectItem[] = [
  { label: 'None', value: 'none' },
  { label: 'v_prediction', value: 'v_prediction' },
  { label: 'epsilon', value: 'epsilon' },
  { label: 'sample', value: 'sample' },
];

type ExtendedImportModelConfig = {
  location: string;
  prediction_type?: 'v_prediction' | 'epsilon' | 'sample' | 'none' | undefined;
};

export default function SimpleAddModels() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();

  const addModelForm = useForm<ExtendedImportModelConfig>({
    initialValues: {
      location: '',
      prediction_type: undefined,
    },
  });

  const handleAddModelSubmit = (values: ExtendedImportModelConfig) => {
    const importModelResponseBody = {
      location: values.location,
      prediction_type:
        values.prediction_type === 'none' ? undefined : values.prediction_type,
    };

    importMainModel({ body: importModelResponseBody })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.modelAddSimple'),
              status: 'success',
            })
          )
        );
        addModelForm.reset();
      })
      .catch((error) => {
        if (error) {
          console.log(error);
          dispatch(
            addToast(
              makeToast({
                title: `${error.data.detail} `,
                status: 'error',
              })
            )
          );
        }
      });
  };

  return (
    <form
      onSubmit={addModelForm.onSubmit((v) => handleAddModelSubmit(v))}
      style={{ width: '100%' }}
    >
      <Flex flexDirection="column" width="100%" gap={4}>
        <IAIMantineTextInput
          label={t('modelManager.modelLocation')}
          placeholder={t('modelManager.simpleModelDesc')}
          w="100%"
          {...addModelForm.getInputProps('location')}
        />
        <IAIMantineSelect
          label={t('modelManager.predictionType')}
          data={predictionSelectData}
          defaultValue="none"
          {...addModelForm.getInputProps('prediction_type')}
        />
        <IAIButton type="submit" isLoading={isLoading}>
          {t('modelManager.addModel')}
        </IAIButton>
      </Flex>
    </form>
  );
}
