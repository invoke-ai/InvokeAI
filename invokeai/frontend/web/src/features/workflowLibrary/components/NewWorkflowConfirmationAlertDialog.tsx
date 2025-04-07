import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseDisclosure } from 'common/hooks/useBoolean';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { useWorkflowLibraryModal } from 'features/nodes/store/workflowLibraryModal';
import { workflowModeChanged } from 'features/nodes/store/workflowLibrarySlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const [useDialogState] = buildUseDisclosure(false);

export const useNewWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dialog = useDialogState();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const workflowLibraryModal = useWorkflowLibraryModal();

  const createImmediate = useCallback(() => {
    dispatch(nodeEditorReset());
    dispatch(workflowModeChanged('edit'));
    workflowLibraryModal.close();

    toast({
      id: 'NEW_WORKFLOW_CREATED',
      title: t('workflows.newWorkflowCreated'),
      status: 'success',
    });

    dialog.close();
  }, [dialog, dispatch, t, workflowLibraryModal]);

  const createWithDialog = useCallback(() => {
    if (!doesWorkflowHaveUnsavedChanges) {
      createImmediate();
      return;
    }
    dialog.open();
  }, [doesWorkflowHaveUnsavedChanges, dialog, createImmediate]);

  return {
    createImmediate,
    createWithDialog,
  } as const;
};

export const NewWorkflowConfirmationAlertDialog = memo(() => {
  useAssertSingleton('NewWorkflowConfirmationAlertDialog');
  const { t } = useTranslation();
  const dialog = useDialogState();
  const newWorkflow = useNewWorkflow();

  return (
    <ConfirmationAlertDialog
      isOpen={dialog.isOpen}
      onClose={dialog.close}
      title={t('nodes.newWorkflow')}
      acceptCallback={newWorkflow.createImmediate}
      useInert={false}
      acceptButtonText={t('common.load')}
    >
      <Flex flexDir="column" gap={2}>
        <Text>{t('nodes.newWorkflowDesc')}</Text>
        <Text variant="subtext">{t('nodes.newWorkflowDesc2')}</Text>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

NewWorkflowConfirmationAlertDialog.displayName = 'NewWorkflowConfirmationAlertDialog';
