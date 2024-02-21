import { Box, Button, Divider,Flex, FormControl, FormLabel, Heading, Input } from '@invoke-ai/ui-library';
import { useForm } from '@mantine/form';
import { useAppDispatch } from 'app/store/storeHooks';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { t } from 'i18next';
import type { CSSProperties } from 'react';
import { useImportMainModelsMutation } from 'services/api/endpoints/models';

import { ImportQueue } from './ImportQueue';

const formStyles: CSSProperties = {
  width: '100%',
};

type ExtendedImportModelConfig = {
  location: string;
};

export const ImportModels = () => {
  const dispatch = useAppDispatch();

  const [importMainModel, { isLoading }] = useImportMainModelsMutation();

  const addModelForm = useForm({
    initialValues: {
      location: '',
    },
  });

  const handleAddModelSubmit = (values: ExtendedImportModelConfig) => {
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
    <Box layerStyle="first" p={3} borderRadius="base" w="full" h="full">
      <Box w="full" p={4}>
        <Heading fontSize="xl">Add Model</Heading>
      </Box>
      <Box layerStyle="second" p={3} borderRadius="base" w="full" h="full">
        <form onSubmit={addModelForm.onSubmit((v) => handleAddModelSubmit(v))} style={formStyles}>
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
        <Divider mt="5" mb="3" />
        <ImportQueue />
      </Box>
    </Box>
  );
};
