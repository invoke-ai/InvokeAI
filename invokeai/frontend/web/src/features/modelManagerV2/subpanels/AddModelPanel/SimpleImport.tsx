import { Flex, FormControl, FormLabel, Input, Button } from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { useForm } from '@mantine/form';
import { useAppDispatch } from '../../../../app/store/storeHooks';
import { useImportMainModelsMutation } from '../../../../services/api/endpoints/models';
import { addToast } from '../../../system/store/systemSlice';
import { makeToast } from '../../../system/util/makeToast';

type SimpleImportModelConfig = {
  location: string;
};

export const SimpleImport = () => {
  const dispatch = useAppDispatch();

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();

  const addModelForm = useForm({
    initialValues: {
      location: '',
    },
  });

  const handleAddModelSubmit = (values: SimpleImportModelConfig) => {
    importMainModel({ source: values.location, config: undefined })
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
    <form onSubmit={addModelForm.onSubmit((v) => handleAddModelSubmit(v))}>
      <Flex gap={2} alignItems="flex-end" justifyContent="space-between">
        <FormControl>
          <Flex direction="column" w="full">
            <FormLabel>{t('modelManager.modelLocation')}</FormLabel>
            <Input {...addModelForm.getInputProps('location')} />
          </Flex>
        </FormControl>
        <Button isDisabled={!addModelForm.values.location} isLoading={isLoading} type="submit">
          {t('modelManager.addModel')}
        </Button>
      </Flex>
    </form>
  );
};
