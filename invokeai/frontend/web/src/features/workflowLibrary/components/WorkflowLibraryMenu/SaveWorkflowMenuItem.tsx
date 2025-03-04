import { MenuItem } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectWorkflowIsTouched } from 'features/nodes/store/workflowSlice';
import { useSaveOrSaveAsWorkflow } from 'features/workflowLibrary/hooks/useSaveOrSaveAsWorkflow';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

const SaveWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const saveOrSaveAsWorkflow = useSaveOrSaveAsWorkflow();
  const isTouched = useAppSelector(selectWorkflowIsTouched);

  return (
    <MenuItem as="button" isDisabled={!isTouched} icon={<PiFloppyDiskBold />} onClick={saveOrSaveAsWorkflow}>
      {t('workflows.saveWorkflow')}
    </MenuItem>
  );
};

export default memo(SaveWorkflowMenuItem);
