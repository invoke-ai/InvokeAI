import {
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
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppSelector } from 'app/store/storeHooks';
import type { SubmitHandler, UseFormReturn } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { UpdateModelArg } from 'services/api/endpoints/models';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';

import BaseModelSelect from './Fields/BaseModelSelect';
import ModelVariantSelect from './Fields/ModelVariantSelect';
import PredictionTypeSelect from './Fields/PredictionTypeSelect';

type Props = {
  form: UseFormReturn<UpdateModelArg['body']>;
  onSubmit: SubmitHandler<UpdateModelArg['body']>;
};

const stringFieldOptions = {
  validate: (value?: string | null) => (value && value.trim().length > 3) || 'Must be at least 3 characters',
};

export const ModelEdit = ({ form }: Props) => {
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);
  const { t } = useTranslation();

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }

  return (
    <Flex flexDir="column" h="full">
      <form>
        <Flex w="full" justifyContent="space-between" gap={4} alignItems="center">
          <FormControl flexDir="column" alignItems="flex-start" gap={1} isInvalid={Boolean(form.formState.errors.name)}>
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
          <SimpleGrid columns={2} gap={4}>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.baseModel')}</FormLabel>
              <BaseModelSelect control={form.control} />
            </FormControl>
            <FormControl flexDir="column" alignItems="flex-start" gap={1}>
              <FormLabel>{t('modelManager.variant')}</FormLabel>
              <ModelVariantSelect control={form.control} />
            </FormControl>
            {data.type === 'main' && data.format === 'checkpoint' && (
              <>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.pathToConfig')}</FormLabel>
                  <Input {...form.register('config_path', stringFieldOptions)} />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.predictionType')}</FormLabel>
                  <PredictionTypeSelect control={form.control} />
                </FormControl>
                <FormControl flexDir="column" alignItems="flex-start" gap={1}>
                  <FormLabel>{t('modelManager.upcastAttention')}</FormLabel>
                  <Checkbox {...form.register('upcast_attention')} />
                </FormControl>
              </>
            )}
          </SimpleGrid>
        </Flex>
      </form>
    </Flex>
  );
};
