import { Button, Checkbox, Flex, FormControl, FormLabel, Input, Spacer, Textarea } from '@invoke-ai/ui-library';
import { showSystemPromptsList } from 'features/systemPrompts/store/systemPromptModal';
import { toast } from 'features/toast/toast';
import { memo, useCallback, useMemo } from 'react';
import type { SubmitHandler } from 'react-hook-form';
import { useForm } from 'react-hook-form';
import { useTranslation } from 'react-i18next';
import { useGetSetupStatusQuery } from 'services/api/endpoints/auth';
import type { SystemPromptRecordDTO } from 'services/api/endpoints/systemPrompts';
import { useCreateSystemPromptMutation, useUpdateSystemPromptMutation } from 'services/api/endpoints/systemPrompts';

type FormValues = {
  name: string;
  content: string;
  is_public: boolean;
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
  const { data: setupStatus } = useGetSetupStatusQuery();
  const isMultiuser = setupStatus?.multiuser_enabled ?? false;

  const defaultValues = useMemo<FormValues>(
    () => ({
      name: editing?.name ?? '',
      content: editing?.content ?? '',
      is_public: editing?.is_public ?? false,
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
          // Only forward is_public when multiuser is on; otherwise the backend defaults are correct.
          const changes = isMultiuser
            ? { name: data.name, content: data.content, is_public: data.is_public }
            : { name: data.name, content: data.content };
          await updateSystemPrompt({ id: editing.id, changes }).unwrap();
        } else {
          // Create endpoint sets is_public from server (true single-user, false multiuser);
          // sharing a freshly-created prompt happens via a follow-up edit.
          await createSystemPrompt({ name: data.name, content: data.content }).unwrap();
        }
        showSystemPromptsList();
      } catch {
        toast({ status: 'error', title: t('systemPrompts.unableToSavePrompt') });
      }
    },
    [editing, isMultiuser, updateSystemPrompt, createSystemPrompt, t]
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
      {isMultiuser && editing && (
        <FormControl>
          <Checkbox {...register('is_public')}>{t('systemPrompts.shareWithEveryone')}</Checkbox>
        </FormControl>
      )}
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
