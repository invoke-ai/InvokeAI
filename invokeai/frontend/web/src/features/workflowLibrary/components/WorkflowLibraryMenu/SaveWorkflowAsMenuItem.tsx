import { MenuItem, useDisclosure } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyBold } from 'react-icons/pi';

import { SaveWorkflowAsDialog } from './SaveWorkflowAsDialog';

const SaveWorkflowAsButton = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <>
      <MenuItem as="button" icon={<PiCopyBold />} onClick={onOpen}>
        {t('workflows.saveWorkflowAs')}
      </MenuItem>

      <SaveWorkflowAsDialog isOpen={isOpen} onClose={onClose} />
    </>
  );
};

export default memo(SaveWorkflowAsButton);
