import { Button, Flex, Heading, Spacer, Text } from '@invoke-ai/ui-library';
import { skipToken } from '@reduxjs/toolkit/query';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setSelectedModelMode } from 'features/modelManagerV2/store/modelManagerV2Slice';
import { ModelConvertButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelConvertButton';
import { ModelEditButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelEditButton';
import { addToast } from 'features/system/store/systemSlice';
import { makeToast } from 'features/system/util/makeToast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { PiCheckBold, PiXBold } from 'react-icons/pi';
import type { UpdateModelArg } from 'services/api/endpoints/models';
import { useGetModelConfigQuery, useUpdateModelMutation } from 'services/api/endpoints/models';

import ModelImageUpload from './Fields/ModelImageUpload';
import { ModelEdit } from './ModelEdit';
import { ModelView } from './ModelView';

export const Model = () => {
  const { t } = useTranslation();
  const selectedModelMode = useAppSelector((s) => s.modelmanagerV2.selectedModelMode);
  const selectedModelKey = useAppSelector((s) => s.modelmanagerV2.selectedModelKey);
  const { data, isLoading } = useGetModelConfigQuery(selectedModelKey ?? skipToken);
  const [updateModel, { isLoading: isSubmitting }] = useUpdateModelMutation();
  const dispatch = useAppDispatch();

  const form = useForm<UpdateModelArg['body']>({
    defaultValues: data,
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<UpdateModelArg['body']>>(
    (values) => {
      if (!data?.key) {
        return;
      }

      const responseBody: UpdateModelArg = {
        key: data.key,
        body: values,
      };

      updateModel(responseBody)
        .unwrap()
        .then((payload) => {
          form.reset(payload, { keepDefaultValues: true });
          dispatch(setSelectedModelMode('view'));
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
          form.reset();
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
    [dispatch, data?.key, form, t, updateModel]
  );

  const handleClickCancel = useCallback(() => {
    dispatch(setSelectedModelMode('view'));
  }, [dispatch]);

  if (isLoading) {
    return <Text>{t('common.loading')}</Text>;
  }

  if (!data) {
    return <Text>{t('common.somethingWentWrong')}</Text>;
  }

  return (
    <Flex flexDir="column" gap={4}>
      <Flex alignItems="flex-start" gap={4}>
        <ModelImageUpload model_key={selectedModelKey} model_image={data.cover_image} />
        <Flex flexDir="column" gap={1} flexGrow={1} minW={0}>
          <Flex gap={2}>
            <Heading as="h2" fontSize="lg" noOfLines={1} wordBreak="break-all">
              {data.name}
            </Heading>
            <Spacer />
            {selectedModelMode === 'view' && <ModelConvertButton modelKey={selectedModelKey} />}
            {selectedModelMode === 'view' && <ModelEditButton />}
            {selectedModelMode === 'edit' && (
              <Button size="sm" onClick={handleClickCancel} leftIcon={<PiXBold />}>
                {t('common.cancel')}
              </Button>
            )}
            {selectedModelMode === 'edit' && (
              <Button
                size="sm"
                colorScheme="invokeYellow"
                leftIcon={<PiCheckBold />}
                onClick={form.handleSubmit(onSubmit)}
                isLoading={isSubmitting}
                isDisabled={Boolean(Object.keys(form.formState.errors).length)}
              >
                {t('common.save')}
              </Button>
            )}
          </Flex>
          {data.source && (
            <Text variant="subtext" noOfLines={1} wordBreak="break-all">
              {t('modelManager.source')}: {data?.source}
            </Text>
          )}
          <Text noOfLines={3}>{data.description}</Text>
        </Flex>
      </Flex>
      {selectedModelMode === 'view' ? <ModelView /> : <ModelEdit form={form} onSubmit={onSubmit} />}
    </Flex>
  );
};
