import { IconButton } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowIsTouched } from 'features/nodes/store/workflowSlice';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowButton = () => {
  const { t } = useTranslation();
  const isTouched = useAppSelector(selectWorkflowIsTouched);
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();

  return (
    <IconButton
      tooltip={t('workflows.saveWorkflow')}
      aria-label={t('workflows.saveWorkflow')}
      icon={<PiFloppyDiskBold />}
      isDisabled={!isTouched}
      onClick={saveOrSaveAsWorkflow}
      pointerEvents="auto"
    />
  );
};

export default memo(SaveWorkflowButton);
