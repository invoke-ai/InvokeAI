import { MenuItem } from '@invoke-ai/ui-library';
import { useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveLibraryWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const { saveWorkflow } = useSaveLibraryWorkflow();
  return (
    <MenuItem as="button" icon={<PiFloppyDiskBold />} onClick={saveWorkflow}>
      {t('workflows.saveWorkflow')}
    </MenuItem>
  );
};

export default memo(SaveLibraryWorkflowMenuItem);
