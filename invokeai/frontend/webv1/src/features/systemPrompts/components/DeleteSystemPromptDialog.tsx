import { ConfirmationAlertDialog, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { toast } from 'features/toast/toast';
import { atom } from 'nanostores';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import type { SystemPromptRecordDTO } from 'services/api/endpoints/systemPrompts';
import { useDeleteSystemPromptMutation } from 'services/api/endpoints/systemPrompts';

const $promptToDelete = atom<SystemPromptRecordDTO | null>(null);
const clearPromptToDelete = () => $promptToDelete.set(null);

export const useDeleteSystemPrompt = () => {
  return useCallback((prompt: SystemPromptRecordDTO) => {
    $promptToDelete.set(prompt);
  }, []);
};

export const DeleteSystemPromptDialog = memo(() => {
  useAssertSingleton('DeleteSystemPromptDialog');
  const { t } = useTranslation();
  const promptToDelete = useStore($promptToDelete);
  const [_deleteSystemPrompt] = useDeleteSystemPromptMutation();

  const deleteSystemPrompt = useCallback(async () => {
    if (!promptToDelete) {
      return;
    }
    try {
      await _deleteSystemPrompt(promptToDelete.id).unwrap();
      toast({ status: 'success', title: t('systemPrompts.promptDeleted') });
    } catch {
      toast({ status: 'error', title: t('systemPrompts.unableToDeletePrompt') });
    }
  }, [promptToDelete, _deleteSystemPrompt, t]);

  return (
    <ConfirmationAlertDialog
      isOpen={promptToDelete !== null}
      onClose={clearPromptToDelete}
      title={t('systemPrompts.deletePrompt')}
      acceptCallback={deleteSystemPrompt}
      acceptButtonText={t('common.delete')}
      cancelButtonText={t('common.cancel')}
      useInert={false}
    >
      <Text>{t('systemPrompts.deletePromptConfirm')}</Text>
    </ConfirmationAlertDialog>
  );
});

DeleteSystemPromptDialog.displayName = 'DeleteSystemPromptDialog';
