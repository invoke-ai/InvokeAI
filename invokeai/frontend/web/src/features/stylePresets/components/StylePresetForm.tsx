import { Button, Flex, FormControl, FormLabel, Input, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { isModalOpenChanged, updatingStylePresetIdChanged } from 'features/stylePresets/store/stylePresetModalSlice';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useCreateStylePresetMutation, useUpdateStylePresetMutation } from 'services/api/endpoints/stylePresets';

import { StylePresetImageField } from './StylePresetImageField';
import { StylePresetPromptField } from './StylePresetPromptField';

export type StylePresetFormData = {
  name: string;
  positivePrompt: string;
  negativePrompt: string;
  image: File | null;
};

export const StylePresetForm = ({ updatingStylePresetId }: { updatingStylePresetId: string | null }) => {
  const [createStylePreset] = useCreateStylePresetMutation();
  const [updateStylePreset] = useUpdateStylePresetMutation();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const defaultValues = useAppSelector((s) => s.stylePresetModal.prefilledFormData);

  const { handleSubmit, control, register } = useForm<StylePresetFormData>({
    defaultValues: defaultValues || {
      name: '',
      positivePrompt: '',
      negativePrompt: '',
      image: null,
    },
  });

  const handleClickSave = useCallback<SubmitHandler<StylePresetFormData>>(
    async (data) => {
      const payload = {
        name: data.name,
        positive_prompt: data.positivePrompt,
        negative_prompt: data.negativePrompt,
        image: data.image,
      };

      try {
        if (updatingStylePresetId) {
          await updateStylePreset({
            id: updatingStylePresetId,
            ...payload,
          }).unwrap();
        } else {
          await createStylePreset(payload).unwrap();
        }
      } catch (error) {
        toast({
          status: 'error',
          title: 'Failed to save style preset',
        });
      }

      dispatch(updatingStylePresetIdChanged(null));
      dispatch(isModalOpenChanged(false));
    },
    [dispatch, updatingStylePresetId, updateStylePreset, createStylePreset]
  );

  return (
    <Flex flexDir="column" gap="4">
      <Flex alignItems="center" gap="4">
        <StylePresetImageField control={control} name="image" />
        <FormControl orientation="vertical">
          <FormLabel>{t('stylePresets.name')}</FormLabel>
          <Input size="md" {...register('name')} />
        </FormControl>
      </Flex>

      <StylePresetPromptField label="Positive Prompt" control={control} name="positivePrompt" />
      <StylePresetPromptField label="Negative Prompt" control={control} name="negativePrompt" />
      <Text variant="subtext">{t('stylePresets.placeholderDirections')}</Text>

      <Flex justifyContent="flex-end">
        <Button onClick={handleSubmit(handleClickSave)}>{t('common.save')}</Button>
      </Flex>
    </Flex>
  );
};
