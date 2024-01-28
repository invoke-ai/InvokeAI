import { Button, useDisclosure } from '@invoke-ai/ui-library';
import { WorkflowLibraryModalContext } from 'features/workflowLibrary/context/WorkflowLibraryModalContext';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBooksBold } from 'react-icons/pi';

import WorkflowLibraryModal from './WorkflowLibraryModal';

const WorkflowLibraryButton = () => {
  const { t } = useTranslation();
  const disclosure = useDisclosure();

  return (
    <WorkflowLibraryModalContext.Provider value={disclosure}>
      <Button leftIcon={<PiBooksBold />} onClick={disclosure.onOpen} pointerEvents="auto">
        {t('workflows.workflowLibrary')}
      </Button>
      <WorkflowLibraryModal />
    </WorkflowLibraryModalContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
