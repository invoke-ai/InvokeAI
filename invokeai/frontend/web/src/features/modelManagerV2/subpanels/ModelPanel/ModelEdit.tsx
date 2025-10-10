import {
  Button,
  Checkbox,
  Flex,
  FormControl,
  FormErrorMessage,
  FormLabel,
  Heading,
  Input,
  SimpleGrid,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { ModelHeader } from 'features/modelManagerV2/subpanels/ModelPanel/ModelHeader';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { type SubmitHandler, useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiXBold } from 'react-icons/pi';
import { type UpdateModelArg, useUpdateModelMutation } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

import BaseModelSelect from './Fields/BaseModelSelect';
import ModelFormatSelect from './Fields/ModelFormatSelect';
import ModelTypeSelect from './Fields/ModelTypeSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';
import { ModelFooter } from './ModelFooter';

type Props = {
  modelConfig: AnyModelConfig;
};

const stringFieldOptions = {
  validate: (value?: string | null) => (value && value.trim().length > 3) || 'Must be at least 3 characters',
};

export const ModelEdit = memo(({ modelConfig }: Props) => {
  const { t } = useTranslation();
  const [updateModel, { isLoading: isSubmitting }] = useUpdateModelMutation();
  const dispatch = useAppDispatch();

  const form = useForm<UpdateModelArg['body']>({
    defaultValues: modelConfig,
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<UpdateModelArg['body']>>(
    (values) => {
      const responseBody: UpdateModelArg = {
        key: modelConfig.key,
        body: values,
      };

      updateModel(responseBody)
        .unwrap()
        .then((payload) => {
          form.reset(payload, { keepDefaultValues: true });
          dispatch(setSelectedModelMode('view'));
          toast({
            id: 'MODEL_UPDATED',
            title: t('modelManager.modelUpdated'),
            status: 'success',
          });
        })
        .catch((_) => {
          form.reset();
          toast({
            id: 'MODEL_UPDATE_FAILED',
            title: t('modelManager.modelUpdateFailed'),
            status: 'error',
          });
        });
    },
    [dispatch, modelConfig.key, form, t, updateModel]
  );

  const handleClickCancel = useCallback(() => {
    dispatch(setSelectedModelMode('view'));
  }, [dispatch]);

  return (
    <Flex flexDir="column" gap={4}>
      <ModelHeader modelConfig={modelConfig}>
        <Button flexShrink={0} size="sm" onClick={handleClickCancel} leftIcon={<PiXBold />}>
          {t('common.cancel')}
        </Button>
        <Button
          flexShrink={0}
          size="sm"
          colorScheme="invokeYellow"
          leftIcon={<PiCheckBold />}
          onClick={form.handleSubmit(onSubmit)}
          isLoading={isSubmitting}
          isDisabled={Boolean(Object.keys(form.formState.errors).length)}
        >
          {t('common.save')}
        </Button>
      </ModelHeader>
      <Flex flexDir="column" h="full">
        <form>
          <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
            <FormControl
              flexDir="column"
              alignItems="flex-start"
              gap={1}
              isInvalid={Boolean(form.formState.errors.name)}
            >
              <FormLabel>{t('modelManager.modelName')}</FormLabel>
              <Input {...form.register('name', stringFieldOptions)} size="md" />

              {form.formState.errors.name?.message && (
                <FormErrorMessage>{form.formState.errors.name?.message}</FormErrorMessage>
              )}
            </FormControl>
          </Flex>

          <Flex flexDir="column" gap={3} mt="4">
            <Flex gap="4" alignItems="center">
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.description')}</FormLabel>
                <Textarea {...form.register('description')} minH={32} />
              </FormControl>
            </Flex>
            <Heading as="h3" fontSize="md" mt="4">
              {t('modelManager.modelSettings')}
            </Heading>
            <Text variant="subtext" color="warning.300">
              Careful! Change these settings only if Invoke didn&apos;t detect them correctly when you installed the
              model. If you choose the wrong settings, the model may not work properly.
            </Text>
            <SimpleGrid columns={2} gap={4}>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.modelType')}</FormLabel>
                <ModelTypeSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.modelFormat')}</FormLabel>
                <ModelFormatSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.baseModel')}</FormLabel>
                <BaseModelSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.variant')}</FormLabel>
                <ModelVariantSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.pathToConfig')}</FormLabel>
                <Input {...form.register('config_path')} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.predictionType')}</FormLabel>
                <PredictionTypeSelect control={form.control} />
              </FormControl>
              <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                <FormLabel>{t('modelManager.upcastAttention')}</FormLabel>
                <Checkbox {...form.register('upcast_attention')} />
              </FormControl>
            </SimpleGrid>
          </Flex>
        </form>
      </Flex>
      <ModelFooter modelConfig={modelConfig} isEditing={true} />
    </Flex>
  );
});

ModelEdit.displayName = 'ModelEdit';
