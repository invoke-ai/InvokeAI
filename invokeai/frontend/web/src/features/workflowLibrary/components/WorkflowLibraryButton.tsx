import { IconButton, useDisclosure } from '@invoke-ai/ui-library';
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
      <IconButton
        aria-label={t('workflows.workflowLibrary')}
        tooltip={t('workflows.workflowLibrary')}
        icon={<PiBooksBold />}
        onClick={disclosure.onOpen}
        pointerEvents="auto"
      />
      <WorkflowLibraryModal />
    </WorkflowLibraryModalContext.Provider>
  );
};

export default memo(WorkflowLibraryButton);
