import type { ComboboxOption } from '@invoke-ai/ui-library';
import { Button, Combobox, Flex, FormControl, FormLabel, Input } from '@invoke-ai/ui-library';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import type { CSSProperties } from 'react';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useImportMainModelsMutation } from 'services/api/endpoints/models';

const options: ComboboxOption[] = [
  { label: 'None', value: 'none' },
  { label: 'v_prediction', value: 'v_prediction' },
  { label: 'epsilon', value: 'epsilon' },
  { label: 'sample', value: 'sample' },
];

type ExtendedImportModelConfig = {
  location: string;
  prediction_type?: 'v_prediction' | 'epsilon' | 'sample' | 'none' | undefined;
};

const SimpleAddModels = () => {
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
      prediction_type: values.prediction_type === 'none' ? undefined : values.prediction_type,
    };

    importMainModel({ body: importModelResponseBody })
      .unwrap()
      .then((_) => {
        dispatch(
          addToast(
            makeToast({
              title: t('toast.modelAddedSimple'),
              status: 'success',
            })
          )
        );
        addModelForm.reset();
      })
      .catch((error) => {
        if (error) {
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
    <form onSubmit={addModelForm.onSubmit((v) => handleAddModelSubmit(v))} style={formStyles}>
      <Flex flexDirection="column" width="100%" gap={4}>
        <FormControl>
          <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
          <Input placeholder={t('modelManager.simpleModelDesc')} w="100%" {...addModelForm.getInputProps('location')} />
        </FormControl>
        <FormControl>
          <FormLabel>{t('modelManager.predictionType')}</FormLabel>
          <Combobox options={options} defaultValue={options[0]} {...addModelForm.getInputProps('prediction_type')} />
        </FormControl>
        <Button type="submit" isLoading={isLoading}>
          {t('modelManager.addModel')}
        </Button>
      </Flex>
    </form>
  );
};

const formStyles: CSSProperties = {
  width: '100%',
};

export default memo(SimpleAddModels);
