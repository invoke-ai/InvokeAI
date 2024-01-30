import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

import { useSaveLibraryWorkflow } from '../../../../../workflowLibrary/hooks/useSaveWorkflow';

const SaveWorkflowButton = () => {
  const { t } = useTranslation();
  const isTouched = useAppSelector((s) => s.workflow.isTouched);

  const { saveWorkflow } = useSaveLibraryWorkflow();

  return (
    <IconButton
      tooltip={t('workflows.saveWorkflow')}
      aria-label={t('workflows.saveWorkflow')}
      icon={<PiFloppyDiskBold />}
      isDisabled={!isTouched}
      onClick={saveWorkflow}
      pointerEvents="auto"
    />
  );
};

export default memo(SaveWorkflowButton);
