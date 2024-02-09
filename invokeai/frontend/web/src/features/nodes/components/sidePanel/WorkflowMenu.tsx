import { Button, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { $builtWorkflow } from 'features/nodes/hooks/useWorkflowWatcher';
import { selectWorkflowSlice } from 'features/nodes/store/workflowSlice';
import NewWorkflowButton from 'features/workflowLibrary/components/NewWorkflowButton';
import { useSaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog/useSaveWorkflowAsDialog';
import UploadWorkflowButton from 'features/workflowLibrary/components/UploadWorkflowButton';
import WorkflowLibraryButton from 'features/workflowLibrary/components/WorkflowLibraryButton';
import { isWorkflowWithID, useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createMemoizedSelector(selectWorkflowSlice, (workflow) => {
  return {
    mode: workflow.mode,
    isTouched: workflow.isTouched,
  };
});

export const WorkflowMenu = () => {
  const { mode, isTouched } = useAppSelector(selector);
  const { onOpen } = useSaveWorkflowAsDialog();
  const { saveWorkflow } = useSaveLibraryWorkflow();
  const { t } = useTranslation();

  const handleClickSave = useCallback(async () => {
    const builtWorkflow = $builtWorkflow.get();
    if (!builtWorkflow) {
      return;
    }

    if (isWorkflowWithID(builtWorkflow)) {
      saveWorkflow();
    } else {
      onOpen();
    }
  }, [onOpen, saveWorkflow]);

  return (
    <Flex gap="2" alignItems="center">
      {mode === 'edit' && isTouched && (
        <Button size="xs" variant="link" onClick={handleClickSave}>
          {t('workflows.saveWorkflow')}
        </Button>
      )}
      <WorkflowLibraryButton />
      <UploadWorkflowButton />
      <NewWorkflowButton />
    </Flex>
  );
};
