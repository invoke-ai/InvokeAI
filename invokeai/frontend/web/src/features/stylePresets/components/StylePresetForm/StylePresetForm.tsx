import { Box, Button, Flex, FormControl, FormLabel, Input, Spacer, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { PRESET_PLACEHOLDER } from 'features/stylePresets/hooks/usePresetModifiedPrompts';
import { $stylePresetModalState } from 'features/stylePresets/store/stylePresetModal';
import { selectAllowPrivateStylePresets } from 'features/system/store/configSlice';
import { toast } from 'features/toast/toast';
import type { PropsWithChildren } from 'react';
import { useCallback } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { Trans, useTranslation } from 'react-i18next';
import type { PresetType } from 'services/api/endpoints/stylePresets';
import { useCreateStylePresetMutation, useUpdateStylePresetMutation } from 'services/api/endpoints/stylePresets';

import { StylePresetImageField } from './StylePresetImageField';
import { StylePresetPromptField } from './StylePresetPromptField';
import { StylePresetTypeField } from './StylePresetTypeField';

export type StylePresetFormData = {
  name: string;
  positivePrompt: string;
  negativePrompt: string;
  image: File | null;
  type: PresetType;
};

export const StylePresetForm = ({
  updatingStylePresetId,
  formData,
}: {
  updatingStylePresetId: string | null;
  formData: StylePresetFormData | null;
}) => {
  const [createStylePreset, { isLoading: isCreating }] = useCreateStylePresetMutation();
  const [updateStylePreset, { isLoading: isUpdating }] = useUpdateStylePresetMutation();
  const { t } = useTranslation();
  const allowPrivateStylePresets = useAppSelector(selectAllowPrivateStylePresets);

  const { handleSubmit, control, register, formState } = useForm<StylePresetFormData>({
    defaultValues: formData || {
      name: '',
      positivePrompt: '',
      negativePrompt: '',
      image: null,
      type: 'user',
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
          type: data.type,
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

      <StylePresetPromptField label={t('stylePresets.positivePrompt')} control={control} name="positivePrompt" />
      <StylePresetPromptField label={t('stylePresets.negativePrompt')} control={control} name="negativePrompt" />
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

      <Flex justifyContent="space-between" alignItems="flex-end" gap={10}>
        {allowPrivateStylePresets ? <StylePresetTypeField control={control} name="type" /> : <Spacer />}
        <Button
          onClick={handleSubmit(handleClickSave)}
          isDisabled={!formState.isValid}
          isLoading={isCreating || isUpdating}
        >
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
