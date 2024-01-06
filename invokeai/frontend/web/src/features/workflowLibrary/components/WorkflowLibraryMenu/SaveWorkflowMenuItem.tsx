import { InvMenuItem } from 'common/components/InvMenu/InvMenuItem';
import { useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi'

const SaveLibraryWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const { saveWorkflow } = useSaveLibraryWorkflow();
  return (
    <InvMenuItem as="button" icon={<PiFloppyDiskBold />} onClick={saveWorkflow}>
      {t('workflows.saveWorkflow')}
    </InvMenuItem>
  );
};

export default memo(SaveLibraryWorkflowMenuItem);
