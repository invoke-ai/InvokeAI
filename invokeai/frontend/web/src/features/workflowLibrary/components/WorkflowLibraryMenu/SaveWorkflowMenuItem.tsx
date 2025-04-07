import { MenuItem } from '@invoke-ai/ui-library';
import { useDoesWorkflowHaveUnsavedChanges } from 'features/nodes/components/sidePanel/workflow/IsolatedWorkflowBuilderWatcher';
import { useIsWorkflowPublished } from 'features/nodes/components/sidePanel/workflow/publish';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();
  const doesWorkflowHaveUnsavedChanges = useDoesWorkflowHaveUnsavedChanges();
  const isPublished = useIsWorkflowPublished();

  return (
    <MenuItem
      as="button"
      isDisabled={!doesWorkflowHaveUnsavedChanges || !!isPublished}
      icon={<PiFloppyDiskBold />}
      onClick={saveOrSaveAsWorkflow}
    >
      {t('workflows.saveWorkflow')}
    </MenuItem>
  );
};

export default memo(SaveWorkflowMenuItem);
