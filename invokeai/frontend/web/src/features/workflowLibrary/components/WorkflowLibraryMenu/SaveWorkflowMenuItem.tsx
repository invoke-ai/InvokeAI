import { MenuItem, useDisclosure } from '@invoke-ai/ui-library';
import { isWorkflowWithID, useSaveLibraryWorkflow } from 'features/workflowLibrary/hooks/useSaveWorkflow';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiFloppyDiskBold } from 'react-icons/pi';

import { useAppSelector } from '../../../../app/store/storeHooks';
import { $builtWorkflow } from '../../../nodes/hooks/useWorkflowWatcher';
import { SaveWorkflowAsDialog } from './SaveWorkflowAsDialog';

const SaveLibraryWorkflowMenuItem = () => {
  const { t } = useTranslation();
  const { saveWorkflow } = useSaveLibraryWorkflow();
  const { isOpen, onOpen, onClose } = useDisclosure();
  const isTouched = useAppSelector((s) => s.workflow.isTouched);

  const handleClickSave = useCallback(async () => {
    const builtWorkflow = $builtWorkflow.get();
    if (!builtWorkflow) {
      return;
    }

    if (isWorkflowWithID(builtWorkflow)) {
      saveWorkflow();
    } else {
      onOpen();
    }
  }, [onOpen, saveWorkflow]);

  return (
    <>
      <MenuItem as="button" isDisabled={!isTouched} icon={<PiFloppyDiskBold />} onClick={handleClickSave}>
        {t('workflows.saveWorkflow')}
      </MenuItem>

      <SaveWorkflowAsDialog isOpen={isOpen} onClose={onClose} />
    </>
  );
};

export default memo(SaveLibraryWorkflowMenuItem);
