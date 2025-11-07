import { MenuItem } from '@invoke-ai/ui-library';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();

  return (
    <MenuItem
      as="button"
      isDisabled={!doesWorkflowHaveUnsavedChanges}
      icon={<PiFloppyDiskBold />}
      onClick={saveOrSaveAsWorkflow}
    >
      {t('workflows.saveWorkflow')}
    </MenuItem>
  );
};

export default memo(SaveWorkflowMenuItem);
