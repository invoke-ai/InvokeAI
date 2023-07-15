import { Flex } from '@chakra-ui/react';
// import { addNewModel } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useTranslation } from 'react-i18next';

import { SelectItem } from '@mantine/core';
import { useForm } from '@mantine/form';
import { makeToast } from 'app/components/Toaster';
import { RootState } from 'app/store/store';
import IAIButton from 'common/components/IAIButton';
import IAIMantineTextInput from 'common/components/IAIMantineInput';
import IAIMantineSelect from 'common/components/IAIMantineSelect';
import { addToast } from 'features/system/store/systemSlice';
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

export default function AddModels() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

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
              title: 'Model Added',
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
    <form onSubmit={addModelForm.onSubmit((v) => handleAddModelSubmit(v))}>
      <Flex
        flexDirection="column"
        overflow="scroll"
        maxHeight={window.innerHeight - 270}
        width="100%"
        gap={4}
      >
        <IAIMantineTextInput
          label="Model Location"
          w="100%"
          {...addModelForm.getInputProps('location')}
        />
        <IAIMantineSelect
          label="Prediction Type (used for Stable Diffusion 2.x Models)"
          data={predictionSelectData}
          defaultValue="none"
          {...addModelForm.getInputProps('prediction_type')}
        />
        <IAIButton
          type="submit"
          isLoading={isLoading}
          isDisabled={isLoading || isProcessing}
        >
          {t('modelManager.addModel')}
        </IAIButton>
      </Flex>
    </form>
  );
}
