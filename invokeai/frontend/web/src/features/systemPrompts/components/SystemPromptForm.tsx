import { Button, Flex, FormControl, FormLabel, Input, Spacer, Textarea } from '@invoke-ai/ui-library';
import { showSystemPromptsList } from 'features/systemPrompts/store/systemPromptModal';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import type { SystemPromptRecordDTO } from 'services/api/endpoints/systemPrompts';
import { useCreateSystemPromptMutation, useUpdateSystemPromptMutation } from 'services/api/endpoints/systemPrompts';

type FormValues = {
  name: string;
  content: string;
};

type Props = {
  /**
   * The prompt to edit, or `null` when creating a new one.
   */
  editing: SystemPromptRecordDTO | null;
};

export const SystemPromptForm = memo(({ editing }: Props) => {
  const { t } = useTranslation();
  const [createSystemPrompt, { isLoading: isCreating }] = useCreateSystemPromptMutation();
  const [updateSystemPrompt, { isLoading: isUpdating }] = useUpdateSystemPromptMutation();

  const defaultValues = useMemo<FormValues>(
    () => ({
      name: editing?.name ?? '',
      content: editing?.content ?? '',
    }),
    [editing]
  );

  const { register, handleSubmit, formState } = useForm<FormValues>({
    defaultValues,
    mode: 'onChange',
  });

  const onSubmit = useCallback<SubmitHandler<FormValues>>(
    async (data) => {
      try {
        if (editing) {
          await updateSystemPrompt({ id: editing.id, changes: data }).unwrap();
        } else {
          await createSystemPrompt(data).unwrap();
        }
        showSystemPromptsList();
      } catch {
        toast({ status: 'error', title: t('systemPrompts.unableToSavePrompt') });
      }
    },
    [editing, updateSystemPrompt, createSystemPrompt, t]
  );

  const handleCancel = useCallback(() => {
    showSystemPromptsList();
  }, []);

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl orientation="vertical">
        <FormLabel>{t('systemPrompts.name')}</FormLabel>
        <Input size="md" {...register('name', { required: true, minLength: 1 })} />
      </FormControl>
      <FormControl orientation="vertical">
        <FormLabel>{t('systemPrompts.content')}</FormLabel>
        <Textarea
          rows={22}
          {...register('content', { required: true, minLength: 1 })}
          placeholder={t('systemPrompts.contentPlaceholder')}
        />
      </FormControl>
      <Flex justifyContent="flex-end" gap={2}>
        <Spacer />
        <Button variant="ghost" onClick={handleCancel}>
          {t('common.cancel')}
        </Button>
        <Button
          colorScheme="invokeBlue"
          onClick={handleSubmit(onSubmit)}
          isDisabled={!formState.isValid}
          isLoading={isCreating || isUpdating}
        >
          {t('common.save')}
        </Button>
      </Flex>
    </Flex>
  );
});

SystemPromptForm.displayName = 'SystemPromptForm';
