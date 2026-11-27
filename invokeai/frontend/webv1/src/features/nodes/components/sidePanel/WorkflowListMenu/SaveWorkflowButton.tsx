import { IconButton } from '@invoke-ai/ui-library';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useIsCurrentWorkflowOwner } from 'features/workflowLibrary/hooks/useIsCurrentWorkflowOwner';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowButton = () => {
  const { t } = useTranslation();
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const isCurrentWorkflowOwner = useIsCurrentWorkflowOwner();

  return (
    <IconButton
      tooltip={t('workflows.saveWorkflow')}
      aria-label={t('workflows.saveWorkflow')}
      icon={<PiFloppyDiskBold />}
      isDisabled={!doesWorkflowHaveUnsavedChanges || !isCurrentWorkflowOwner}
      onClick={saveOrSaveAsWorkflow}
      pointerEvents="auto"
      variant="ghost"
      size="sm"
    />
  );
};

export default memo(SaveWorkflowButton);
