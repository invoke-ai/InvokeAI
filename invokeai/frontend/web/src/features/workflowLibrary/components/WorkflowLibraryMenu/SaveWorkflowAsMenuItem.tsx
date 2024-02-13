import { MenuItem } from '@invoke-ai/ui-library';
import { useSaveWorkflowAsDialog } from 'features/workflowLibrary/components/SaveWorkflowAsDialog/useSaveWorkflowAsDialog';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

const SaveWorkflowAsMenuItem = () => {
  const { t } = useTranslation();
  const { onOpen } = useSaveWorkflowAsDialog();

  return (
    <MenuItem as="button" icon={<PiCopyBold />} onClick={onOpen}>
      {t('workflows.saveWorkflowAs')}
    </MenuItem>
  );
};

export default memo(SaveWorkflowAsMenuItem);
