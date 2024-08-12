import { Box, Button, Flex, FormControl, FormLabel, Input, Text } from '@invoke-ai/ui-library';
import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { toast } from 'features/toast/toast';
import type { PropsWithChildren } from 'react';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { Trans, useTranslation } from 'react-i18next';
import { useCreateStylePresetMutation, useUpdateStylePresetMutation } from 'services/api/endpoints/stylePresets';

import { StylePresetImageField } from './StylePresetImageField';
import { StylePresetPromptField } from './StylePresetPromptField';

export type StylePresetFormData = {
  name: string;
  positivePrompt: string;
  negativePrompt: string;
  image: File | null;
};

export const StylePresetForm = ({
  updatingStylePresetId,
  formData,
}: {
  updatingStylePresetId: string | null;
  formData: StylePresetFormData | null;
}) => {
  const [createStylePreset] = useCreateStylePresetMutation();
  const [updateStylePreset] = useUpdateStylePresetMutation();
  const { t } = useTranslation();

  const { handleSubmit, control, register, formState } = useForm<StylePresetFormData>({
    defaultValues: formData || {
      name: '',
      positivePrompt: '',
      negativePrompt: '',
      image: null,
    },
    mode: 'onChange',
  });

  const handleClickSave = useCallback<SubmitHandler<StylePresetFormData>>(
    async (data) => {
      const payload = {
        data: {
          name: data.name,
          positive_prompt: data.positivePrompt,
          negative_prompt: data.negativePrompt,
        },
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

      $stylePresetModalState.set({
        prefilledFormData: null,
        updatingStylePresetId: null,
        isModalOpen: false,
      });
    },
    [updatingStylePresetId, updateStylePreset, createStylePreset]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <Flex alignItems="center" gap={4}>
        <StylePresetImageField control={control} name="image" />
        <FormControl orientation="vertical">
          <FormLabel>{t('stylePresets.name')}</FormLabel>
          <Input size="md" {...register('name', { required: true, minLength: 1 })} />
        </FormControl>
      </Flex>

      <StylePresetPromptField label="Positive Prompt" control={control} name="positivePrompt" />
      <StylePresetPromptField label="Negative Prompt" control={control} name="negativePrompt" />
      <Box>
        <Text variant="subtext">{t('stylePresets.promptTemplatesDesc1')}</Text>
        <Text variant="subtext">
          <Trans
            i18nKey="stylePresets.promptTemplatesDesc2"
            components={{ Pre: <Pre /> }}
            values={{ placeholder: PRESET_PLACEHOLDER }}
          />
        </Text>
        <Text variant="subtext">{t('stylePresets.promptTemplatesDesc3')}</Text>
      </Box>

      <Flex justifyContent="flex-end">
        <Button onClick={handleSubmit(handleClickSave)} isDisabled={!formState.isValid}>
          {t('common.save')}
        </Button>
      </Flex>
    </Flex>
  );
};

const Pre = (props: PropsWithChildren) => (
  <Text as="span" fontFamily="monospace" fontWeight="semibold">
    {props.children}
  </Text>
);
