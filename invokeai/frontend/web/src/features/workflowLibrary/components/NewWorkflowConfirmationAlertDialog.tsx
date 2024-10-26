import { ConfirmationAlertDialog, Flex, Text } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useAssertSingleton } from 'common/hooks/useAssertSingleton';
import { buildUseDisclosure } from 'common/hooks/useBoolean';
import { nodeEditorReset } from 'features/nodes/store/nodesSlice';
import { selectWorkflowIsTouched, workflowModeChanged } from 'features/nodes/store/workflowSlice';
import { toast } from 'features/toast/toast';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const [useDialogState] = buildUseDisclosure(false);

export const useNewWorkflow = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const dialog = useDialogState();
  const isTouched = useAppSelector(selectWorkflowIsTouched);

  const createImmediate = useCallback(() => {
    dispatch(nodeEditorReset());
    dispatch(workflowModeChanged('edit'));

    toast({
      id: 'NEW_WORKFLOW_CREATED',
      title: t('workflows.newWorkflowCreated'),
      status: 'success',
    });

    dialog.close();
  }, [dialog, dispatch, t]);

  const createWithDialog = useCallback(() => {
    if (!isTouched) {
      createImmediate();
      return;
    }
    dialog.open();
  }, [dialog, createImmediate, isTouched]);

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
    >
      <Flex flexDir="column" gap={2}>
        <Text>{t('nodes.newWorkflowDesc')}</Text>
        <Text variant="subtext">{t('nodes.newWorkflowDesc2')}</Text>
      </Flex>
    </ConfirmationAlertDialog>
  );
});

NewWorkflowConfirmationAlertDialog.displayName = 'NewWorkflowConfirmationAlertDialog';
