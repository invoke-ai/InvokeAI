import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowButton = () => {
  const { t } = useTranslation();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();

  return (
    <IAITooltip label={t('workflows.saveWorkflow')}>
      <IconButton
        aria-label={t('workflows.saveWorkflow')}
        icon={<PiFloppyDiskBold />}
        isDisabled={!doesWorkflowHaveUnsavedChanges}
        onClick={saveOrSaveAsWorkflow}
        pointerEvents="auto"
      />
    </IAITooltip>
  );
};

export default memo(SaveWorkflowButton);
